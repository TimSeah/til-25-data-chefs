import torch
import torch.nn as nn
import numpy as np
import random
import os

# --- Configuration (Adopted from your CNN training script) ---
# Environment specific
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS = 100 # Max steps in one round of the game (rl_manager uses this for normalization)

# Neural Network Hyperparameters for CNNDQN
VIEWCONE_CHANNELS = 8
VIEWCONE_HEIGHT = 7
VIEWCONE_WIDTH = 5
OTHER_FEATURES_SIZE = 4 + 2 + 1 + 1 # Direction (4) + Location (2) + Scout (1) + Step (1)

CNN_OUTPUT_CHANNELS_1 = 16
CNN_OUTPUT_CHANNELS_2 = 32
KERNEL_SIZE_1 = (3, 3)
STRIDE_1 = 1
KERNEL_SIZE_2 = (3, 3)
STRIDE_2 = 1
MLP_HIDDEN_LAYER_1_SIZE = 128
MLP_HIDDEN_LAYER_2_SIZE = 128
OUTPUT_ACTIONS = 5
DROPOUT_RATE = 0.2 # Note: Dropout is active during training. For eval, model.eval() handles it.

# Agent settings
EPSILON_INFERENCE = 0.01 # Small epsilon for some exploration, or 0 for pure exploitation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- CNN-DQN Model (Copied from your training script) ---
class CNNDQN(nn.Module):
    def __init__(self, viewcone_channels, viewcone_height, viewcone_width, other_features_size, mlp_hidden1, mlp_hidden2, num_actions, dropout_rate):
        super(CNNDQN, self).__init__()
        # CNN part for viewcone
        self.conv1 = nn.Conv2d(viewcone_channels, CNN_OUTPUT_CHANNELS_1, kernel_size=KERNEL_SIZE_1, stride=STRIDE_1, padding=1) # Assuming padding=1 for same output size if stride=1
        self.relu_conv1 = nn.ReLU()
        
        # Calculate output dimensions after conv1
        h_out1 = (viewcone_height + 2 * 1 - KERNEL_SIZE_1[0]) // STRIDE_1 + 1
        w_out1 = (viewcone_width + 2 * 1 - KERNEL_SIZE_1[1]) // STRIDE_1 + 1
        
        self.conv2 = nn.Conv2d(CNN_OUTPUT_CHANNELS_1, CNN_OUTPUT_CHANNELS_2, kernel_size=KERNEL_SIZE_2, stride=STRIDE_2, padding=1) # Assuming padding=1
        self.relu_conv2 = nn.ReLU()

        # Calculate output dimensions after conv2
        h_out2 = (h_out1 + 2 * 1 - KERNEL_SIZE_2[0]) // STRIDE_2 + 1
        w_out2 = (w_out1 + 2 * 1 - KERNEL_SIZE_2[1]) // STRIDE_2 + 1

        self.cnn_output_flat_size = CNN_OUTPUT_CHANNELS_2 * h_out2 * w_out2
        
        # MLP part for combined features
        self.fc1_mlp = nn.Linear(self.cnn_output_flat_size + other_features_size, mlp_hidden1)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) # Dropout is handled by model.eval()
        
        self.fc2_mlp = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) # Dropout is handled by model.eval()
        
        self.fc_output = nn.Linear(mlp_hidden2, num_actions)

    def forward(self, viewcone_input, other_features_input):
        # Process viewcone through CNN
        x_cnn = self.relu_conv1(self.conv1(viewcone_input))
        x_cnn = self.relu_conv2(self.conv2(x_cnn))
        x_cnn_flat = x_cnn.view(-1, self.cnn_output_flat_size) # Flatten CNN output
        
        # Concatenate flattened CNN output with other features
        combined_features = torch.cat((x_cnn_flat, other_features_input), dim=1)
        
        # Process combined features through MLP
        x = self.relu_fc1(self.fc1_mlp(combined_features))
        x = self.dropout1(x) # model.eval() will disable this
        x = self.relu_fc2(self.fc2_mlp(x))
        x = self.dropout2(x) # model.eval() will disable this
        return self.fc_output(x)

# --- RL Manager using CNNDQN ---
class RLManager:
    def __init__(self, model_path="training_checkpoint_best_100000_resumed_chkpt.pth"): # Default to the checkpoint file
        self.device = DEVICE
        print(f"RLManager: Using device: {self.device}")

        self.model = CNNDQN(VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH,
                              OTHER_FEATURES_SIZE, MLP_HIDDEN_LAYER_1_SIZE,
                              MLP_HIDDEN_LAYER_2_SIZE, OUTPUT_ACTIONS, DROPOUT_RATE).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                print(f"RLManager: Loading from checkpoint file: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'policy_net_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['policy_net_state_dict'])
                    print(f"RLManager: Loaded policy_net_state_dict from checkpoint {model_path}")
                else:
                    # Fallback for older model files that are just the state_dict
                    print(f"RLManager: 'policy_net_state_dict' not found in checkpoint. Assuming {model_path} is a direct state_dict file.")
                    self.model.load_state_dict(checkpoint) # Checkpoint IS the state_dict
                    print(f"RLManager: Loaded direct state_dict from {model_path}")

            except Exception as e:
                print(f"RLManager: Error loading model from {model_path}: {e}. Using random weights.")
                self.model.apply(self._initialize_weights) # Fallback
        else:
            if model_path:
                 print(f"RLManager: Model path {model_path} not found. Initializing model with random weights.")
            else:
                print("RLManager: No model path provided. Initializing model with random weights.")
            self.model.apply(self._initialize_weights)

        self.model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        return [float((tile_value >> i) & 1) for i in range(VIEWCONE_CHANNELS)]

    def process_observation(self, observation_dict):
        raw_viewcone = observation_dict.get("viewcone", np.zeros((VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.uint8))
        if not isinstance(raw_viewcone, np.ndarray):
            raw_viewcone = np.array(raw_viewcone, dtype=np.uint8)
        
        if raw_viewcone.shape != (VIEWCONE_HEIGHT, VIEWCONE_WIDTH):
            padded_viewcone = np.zeros((VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.uint8)
            h, w = raw_viewcone.shape
            h_min, w_min = min(h, VIEWCONE_HEIGHT), min(w, VIEWCONE_WIDTH)
            padded_viewcone[:h_min, :w_min] = raw_viewcone[:h_min, :w_min]
            raw_viewcone = padded_viewcone

        processed_viewcone_channels_data = np.zeros((VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.float32)
        for r in range(VIEWCONE_HEIGHT):
            for c in range(VIEWCONE_WIDTH):
                tile_value = raw_viewcone[r, c]
                unpacked_features = self._unpack_viewcone_tile(tile_value)
                for channel_idx in range(VIEWCONE_CHANNELS):
                    processed_viewcone_channels_data[channel_idx, r, c] = unpacked_features[channel_idx]
        
        state_viewcone_tensor = torch.from_numpy(processed_viewcone_channels_data).float().unsqueeze(0).to(self.device)

        other_features_list = []
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        direction_one_hot[direction % 4] = 1.0
        other_features_list.extend(direction_one_hot)
        
        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        other_features_list.extend([norm_x, norm_y])
        
        other_features_list.append(float(observation_dict.get("scout", 0)))
        
        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS if MAX_STEPS > 0 else 0.0
        other_features_list.append(norm_step)
        
        state_other_np = np.array(other_features_list, dtype=np.float32)
        state_other_tensor = torch.from_numpy(state_other_np).float().unsqueeze(0).to(self.device)

        if state_other_tensor.shape[1] != OTHER_FEATURES_SIZE:
             print(f"RLManager: Warning: Other features size mismatch. Expected {OTHER_FEATURES_SIZE}, got {state_other_tensor.shape[1]}")

        return state_viewcone_tensor, state_other_tensor

    def rl(self, observation_dict):
        if random.random() < EPSILON_INFERENCE:
            return random.randint(0, OUTPUT_ACTIONS - 1)
        else:
            with torch.no_grad():
                state_viewcone, state_other = self.process_observation(observation_dict)
                q_values = self.model(state_viewcone, state_other)
                action = torch.argmax(q_values, dim=1).item()
                return action

# Example usage (optional, for testing)
if __name__ == '__main__':
    print("RL Manager with CNNDQN Initialized for Inference.")
    
    # --- IMPORTANT: For testing, ensure your checkpoint file exists at this path ---
    # --- or change it to a known model file (either checkpoint or direct state_dict) ---
    TEST_MODEL_PATH = "training_checkpoint_best_100000_resumed_chkpt.pth" 
    # TEST_MODEL_PATH = "my_wargame_cnn_agent_best.pth" # Or test with a direct state_dict file

    if not os.path.exists(TEST_MODEL_PATH):
        print(f"\nWARNING: Test model/checkpoint file '{TEST_MODEL_PATH}' not found.")
        print("The manager will initialize with random weights for the test.")
        print("If you want to test loading, please provide a valid path or create a dummy file.\n")
        manager = RLManager(model_path=None) # Test with random weights
    else:
        manager = RLManager(model_path=TEST_MODEL_PATH)
    
    dummy_obs = {
        "viewcone": np.random.randint(0, 256, size=(VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.uint8),
        "direction": 1,
        "location": [MAP_SIZE_X / 2, MAP_SIZE_Y / 2],
        "scout": 1,
        "step": MAX_STEPS / 2
    }
    
    print("\nTesting process_observation:")
    try:
        vc_tensor, other_tensor = manager.process_observation(dummy_obs)
        print(f"Viewcone tensor shape: {vc_tensor.shape}")
        print(f"Other features tensor shape: {other_tensor.shape}")
    except Exception as e:
        print(f"Error in process_observation: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting rl method (action selection):")
    try:
        action = manager.rl(dummy_obs)
        print(f"Selected action: {action}")
    except Exception as e:
        print(f"Error in rl method: {e}")
        import traceback
        traceback.print_exc()