import torch
import torch.nn as nn
import numpy as np
import random
import os

# --- Configuration (Matching new training script for CNNDQN) ---
# Environment specific
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS = 100 # Assuming MAX_STEPS_PER_EPISODE is MAX_STEPS for inference

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
DROPOUT_RATE = 0.2 # Will be inactive in eval mode, but good for class consistency

# Agent settings
EPSILON_INFERENCE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default model paths
DEFAULT_SCOUT_MODEL_PATH = "scout_learning.pth"
DEFAULT_GUARD_MODEL_PATH = "guard_learning.pth"

# --- CNN Deep Q-Network (CNNDQN) Model (from new training script) ---
class CNNDQN(nn.Module):
    def __init__(self, v_c, v_h, v_w, o_f_s, mlp_h1, mlp_h2, n_a, dr):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(v_c, CNN_OUTPUT_CHANNELS_1, KERNEL_SIZE_1, STRIDE_1, padding=1)
        self.relu_conv1 = nn.ReLU()
        
        h1 = (v_h + 2 * 1 - KERNEL_SIZE_1[0]) // STRIDE_1 + 1
        w1 = (v_w + 2 * 1 - KERNEL_SIZE_1[1]) // STRIDE_1 + 1
        
        self.conv2 = nn.Conv2d(CNN_OUTPUT_CHANNELS_1, CNN_OUTPUT_CHANNELS_2, KERNEL_SIZE_2, STRIDE_2, padding=1)
        self.relu_conv2 = nn.ReLU()

        h2 = (h1 + 2 * 1 - KERNEL_SIZE_2[0]) // STRIDE_2 + 1
        w2 = (w1 + 2 * 1 - KERNEL_SIZE_2[1]) // STRIDE_2 + 1
        
        self.cnn_flat_size = CNN_OUTPUT_CHANNELS_2 * h2 * w2
        
        self.fc1_mlp = nn.Linear(self.cnn_flat_size + o_f_s, mlp_h1)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dr)
        
        self.fc2_mlp = nn.Linear(mlp_h1, mlp_h2)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dr)
        
        self.fc_output = nn.Linear(mlp_h2, n_a)

    def forward(self, vc_in, of_in):
        x = self.relu_conv1(self.conv1(vc_in))
        x = self.relu_conv2(self.conv2(x))
        x_flat = x.view(-1, self.cnn_flat_size)
        combined = torch.cat((x_flat, of_in), dim=1)
        x = self.relu_fc1(self.fc1_mlp(combined))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2_mlp(x))
        x = self.dropout2(x)
        return self.fc_output(x)

# --- RL Agent Manager ---
class RLManager:
    def __init__(self, scout_model_path=DEFAULT_SCOUT_MODEL_PATH, guard_model_path=DEFAULT_GUARD_MODEL_PATH):
        self.device = DEVICE
        print(f"RLManager Using device: {self.device}")

        # Initialize Scout Model
        self.scout_model = CNNDQN(
            VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH,
            OTHER_FEATURES_SIZE,
            MLP_HIDDEN_LAYER_1_SIZE, MLP_HIDDEN_LAYER_2_SIZE,
            OUTPUT_ACTIONS, DROPOUT_RATE
        ).to(self.device)
        self._load_model(self.scout_model, scout_model_path, "Scout")

        # Initialize Guard Model
        self.guard_model = CNNDQN(
            VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH,
            OTHER_FEATURES_SIZE,
            MLP_HIDDEN_LAYER_1_SIZE, MLP_HIDDEN_LAYER_2_SIZE,
            OUTPUT_ACTIONS, DROPOUT_RATE
        ).to(self.device)
        self._load_model(self.guard_model, guard_model_path, "Guard")

    def _load_model(self, model, model_path, role_name):
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained {role_name} model from {model_path}")
            except Exception as e:
                print(f"Error loading {role_name} model from {model_path}: {e}. Using random weights for {role_name}.")
                model.apply(self._initialize_weights)
        else:
            if model_path: print(f"{role_name} model not found at {model_path}. Initializing with random weights for {role_name}.")
            else: print(f"No model path provided for {role_name}. Initializing with random weights for {role_name}.")
            model.apply(self._initialize_weights)
        model.eval()

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        return [float((tile_value >> i) & 1) for i in range(VIEWCONE_CHANNELS)]

    def process_observation(self, observation_dict):
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
        
        other_features_list = []
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        direction_one_hot[direction % 4] = 1.0
        other_features_list.extend(direction_one_hot)

        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        other_features_list.extend([norm_x, norm_y])

        # The "scout" field from observation_dict is crucial for feature engineering.
        # It will be 1.0 if the current agent is a scout, 0.0 otherwise.
        scout_role_feature = float(observation_dict.get("scout", 0))
        other_features_list.append(scout_role_feature)

        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS if MAX_STEPS > 0 else 0.0
        other_features_list.append(norm_step)
        
        processed_other_features = np.array(other_features_list, dtype=np.float32)

        vc_tensor = torch.from_numpy(processed_viewcone).float().unsqueeze(0).to(self.device)
        of_tensor = torch.from_numpy(processed_other_features).float().unsqueeze(0).to(self.device)
        
        return vc_tensor, of_tensor

    def rl(self, observation_dict):
        # Determine current agent's role from observation_dict
        # The "scout" key should be 1 if scout, 0 if guard.
        is_scout = observation_dict.get("scout", 0) == 1 
        
        current_model = self.scout_model if is_scout else self.guard_model
        role_name = "Scout" if is_scout else "Guard"

        if random.random() < EPSILON_INFERENCE:
            # print(f"Agent ({role_name}) taking random action due to epsilon.")
            return random.randint(0, OUTPUT_ACTIONS - 1)
        else:
            with torch.no_grad():
                state_viewcone_tensor, state_other_tensor = self.process_observation(observation_dict)
                # print(f"Agent ({role_name}) using its dedicated model for action.")
                q_values = current_model(state_viewcone_tensor, state_other_tensor)
                action = torch.argmax(q_values, dim=1).item()
                return action
