import torch
import torch.nn as nn
import numpy as np
import random
import os

# --- Configuration (Matching new training script for CNNDQN) ---
# Environment specific
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS = 100 # Corresponds to MAX_STEPS_PER_EPISODE for inference

# Neural Network Hyperparameters for CNNDQN
VIEWCONE_CHANNELS = 8
VIEWCONE_HEIGHT = 7
VIEWCONE_WIDTH = 5
OTHER_FEATURES_SIZE = 4 + 2 + 1 + 1 # direction (4), location (2), scout (1), step (1)

CNN_OUTPUT_CHANNELS_1 = 16
CNN_OUTPUT_CHANNELS_2 = 32
KERNEL_SIZE_1 = (3, 3)
STRIDE_1 = 1
KERNEL_SIZE_2 = (3, 3)
STRIDE_2 = 1
MLP_HIDDEN_LAYER_1_SIZE = 128
MLP_HIDDEN_LAYER_2_SIZE = 128
OUTPUT_ACTIONS = 5
DROPOUT_RATE = 0.2 # Inactive in eval mode, but defined for consistency

# Agent settings
EPSILON_INFERENCE = 0.01 # For minimal exploration during inference, can be 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default model path for the unified agent
DEFAULT_MODEL_PATH = "all_cnn_agent_135500.pth"

# --- CNN Deep Q-Network (CNNDQN) Model (from new training script) ---
class CNNDQN(nn.Module):
    def __init__(self, viewcone_channels, viewcone_height, viewcone_width, other_features_size, mlp_hidden1, mlp_hidden2, num_actions, dropout_rate):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(viewcone_channels, CNN_OUTPUT_CHANNELS_1, kernel_size=KERNEL_SIZE_1, stride=STRIDE_1, padding=1)
        self.relu_conv1 = nn.ReLU()
        h_out1 = (viewcone_height + 2 * 1 - KERNEL_SIZE_1[0]) // STRIDE_1 + 1
        w_out1 = (viewcone_width + 2 * 1 - KERNEL_SIZE_1[1]) // STRIDE_1 + 1
        
        self.conv2 = nn.Conv2d(CNN_OUTPUT_CHANNELS_1, CNN_OUTPUT_CHANNELS_2, kernel_size=KERNEL_SIZE_2, stride=STRIDE_2, padding=1)
        self.relu_conv2 = nn.ReLU()
        h_out2 = (h_out1 + 2 * 1 - KERNEL_SIZE_2[0]) // STRIDE_2 + 1
        w_out2 = (w_out1 + 2 * 1 - KERNEL_SIZE_2[1]) // STRIDE_2 + 1

        self.cnn_output_flat_size = CNN_OUTPUT_CHANNELS_2 * h_out2 * w_out2
        self.fc1_mlp = nn.Linear(self.cnn_output_flat_size + other_features_size, mlp_hidden1)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2_mlp = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_output = nn.Linear(mlp_hidden2, num_actions)

    def forward(self, viewcone_input, other_features_input):
        x_cnn = self.relu_conv1(self.conv1(viewcone_input))
        x_cnn = self.relu_conv2(self.conv2(x_cnn))
        x_cnn_flat = x_cnn.view(-1, self.cnn_output_flat_size)
        combined_features = torch.cat((x_cnn_flat, other_features_input), dim=1)
        x = self.relu_fc1(self.fc1_mlp(combined_features))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2_mlp(x))
        x = self.dropout2(x)
        return self.fc_output(x)

# --- RL Agent Manager ---
class RLManager:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.device = DEVICE
        print(f"RLManager Using device: {self.device}")

        self.model = CNNDQN(
            VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH,
            OTHER_FEATURES_SIZE,
            MLP_HIDDEN_LAYER_1_SIZE, MLP_HIDDEN_LAYER_2_SIZE,
            OUTPUT_ACTIONS, DROPOUT_RATE
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}. Using random weights.")
                self.model.apply(self._initialize_weights)
        else:
            if model_path: print(f"Model not found at {model_path}. Initializing with random weights.")
            else: print("No model path provided. Initializing model with random weights.")
            self.model.apply(self._initialize_weights)

        self.model.eval()  # Set the model to evaluation mode (disables dropout etc.)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        # Unpacks a single tile's integer value into a feature vector of VIEWCONE_CHANNELS length.
        return [float((tile_value >> i) & 1) for i in range(VIEWCONE_CHANNELS)]

    def process_observation(self, observation_dict):
        # Process viewcone
        vc_raw = observation_dict.get("viewcone", np.zeros((VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.uint8))
        if not isinstance(vc_raw, np.ndarray):
            vc_raw = np.array(vc_raw)
        
        if vc_raw.shape != (VIEWCONE_HEIGHT, VIEWCONE_WIDTH):
            padded_vc_raw = np.zeros((VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=vc_raw.dtype)
            h, w = vc_raw.shape
            h_min, w_min = min(h, VIEWCONE_HEIGHT), min(w, VIEWCONE_WIDTH)
            padded_vc_raw[:h_min, :w_min] = vc_raw[:h_min, :w_min]
            vc_raw = padded_vc_raw
            
        processed_viewcone = np.zeros((VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.float32)
        for r in range(VIEWCONE_HEIGHT):
            for c in range(VIEWCONE_WIDTH):
                unpacked_features = self._unpack_viewcone_tile(vc_raw[r, c])
                for ch in range(VIEWCONE_CHANNELS):
                    processed_viewcone[ch, r, c] = unpacked_features[ch]
        
        # Process other features
        other_features_list = []
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        direction_one_hot[direction % 4] = 1.0
        other_features_list.extend(direction_one_hot)

        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        other_features_list.extend([norm_x, norm_y])

        # The "scout" field from observation_dict is crucial.
        # The single model was trained with this feature to differentiate behavior.
        scout_role_feature = float(observation_dict.get("scout", 0))
        other_features_list.append(scout_role_feature)

        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS if MAX_STEPS > 0 else 0.0
        other_features_list.append(norm_step)
        
        processed_other_features = np.array(other_features_list, dtype=np.float32)

        # Convert to tensors
        vc_tensor = torch.from_numpy(processed_viewcone).float().unsqueeze(0).to(self.device)
        of_tensor = torch.from_numpy(processed_other_features).float().unsqueeze(0).to(self.device)
        
        return vc_tensor, of_tensor

    def rl(self, observation_dict):
        """
        Selects an action based on the current observation using the single loaded model.
        The model itself should use the 'scout' feature from the observation to adapt its policy.
        """
        if random.random() < EPSILON_INFERENCE:
            return random.randint(0, OUTPUT_ACTIONS - 1)
        else:
            with torch.no_grad():
                state_viewcone_tensor, state_other_tensor = self.process_observation(observation_dict)
                q_values = self.model(state_viewcone_tensor, state_other_tensor)
                action = torch.argmax(q_values, dim=1).item()
                return action
