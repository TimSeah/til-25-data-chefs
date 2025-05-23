import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# --- Configuration ---
# Neural Network Hyperparameters
INPUT_FEATURES = 288  # Calculated below: 7*5*8 (viewcone) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
HIDDEN_LAYER_3_SIZE = 256 # Restored to match training script
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay

# Game Environment Constants
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS = 100

# Agent settings
EPSILON_INFERENCE = 0.01 # Small epsilon for some exploration even during inference, or 0 for pure exploitation

# --- Deep Q-Network (DQN) Model (corrected to match training architecture) ---
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim): # Added hidden_dim3
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3) # fc3 is now a hidden layer
        self.relu3 = nn.ReLU()                         # ReLU for fc3
        self.fc4 = nn.Linear(hidden_dim3, output_dim)  # fc4 is now the output layer

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x)) # Pass through fc3 and its ReLU
        x = self.fc4(x)             # Output from fc4
        return x

# --- RL Agent ---
class RLManager:
    """
    The Reinforcement Learning Agent.
    It processes observations, uses a DQN to select actions, and can be reset.
    """
    def __init__(self, model_path="agent03_new_100k_eps.pth"):
        """
        Initializes the RL Agent.
        Args:
            model_path (str, optional): Path to a pre-trained model file. Defaults to None (random initialization).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize with the corrected DQN architecture
        self.model = DQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS).to(self.device)

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}. Using random weights.")
                # Fallback to random weights if loading fails
                self.model.apply(self._initialize_weights)
        else:
            print("No model path provided. Initializing model with random weights.")
            self.model.apply(self._initialize_weights)

        self.model.eval()  # Set the model to evaluation mode

    def _initialize_weights(self, m):
        """
        Initializes weights of the neural network layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        """
        Unpacks a single tile's 8-bit integer value from the viewcone into a feature vector.
        - Bits 0-1: Tile type (0:No vision, 1:Empty, 2:Recon, 3:Mission)
        - Bit 2: Scout present
        - Bit 3: Guard present
        - Bit 4: Right wall
        - Bit 5: Bottom wall
        - Bit 6: Left wall
        - Bit 7: Top wall

        Returns a list of 8 binary features.
        """        
        tile_features = []
        tile_features.append(float(tile_value & 0b01)) 
        tile_features.append(float((tile_value & 0b10) >> 1)) 
        for i in range(2, 8): 
            tile_features.append(float((tile_value >> i) & 1))
        
        return tile_features

    def process_observation(self, observation_dict):
        """
        Converts the raw observation dictionary into a flat feature tensor for the DQN.
        Args:
            observation_dict (dict): The observation dictionary from the game.
        Returns:
            torch.Tensor: A flat tensor representing the state.
        """
        processed_features = []

        viewcone = observation_dict.get("viewcone", [])
        viewcone_features = []
        for r in range(7): 
            for c in range(5): 
                tile_value = viewcone[r][c] if r < len(viewcone) and c < len(viewcone[r]) else 0 
                viewcone_features.extend(self._unpack_viewcone_tile(tile_value))
        processed_features.extend(viewcone_features) 

        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4:
            direction_one_hot[direction] = 1.0
        processed_features.extend(direction_one_hot) 

        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        processed_features.extend([norm_x, norm_y]) 

        scout_role = float(observation_dict.get("scout", 0))
        processed_features.append(scout_role) 

        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS if MAX_STEPS > 0 else 0.0
        processed_features.append(norm_step) 
        
        if len(processed_features) != INPUT_FEATURES:
            print(f"Warning: Feature length mismatch. Expected {INPUT_FEATURES}, got {len(processed_features)}")
            if len(processed_features) < INPUT_FEATURES:
                processed_features.extend([0.0] * (INPUT_FEATURES - len(processed_features)))
            else:
                processed_features = processed_features[:INPUT_FEATURES]


        return torch.tensor(processed_features, dtype=torch.float32, device=self.device).unsqueeze(0)

    def rl(self, observation_dict):
        """Selects an action based on the current observation.
        Uses epsilon-greedy for exploration if epsilon > 0, otherwise greedy.
        Args:
            observation_dict (dict): The observation dictionary.
        Returns:
            int: The selected action (0-4). See `rl/README.md` for the options.
        """
        if random.random() < EPSILON_INFERENCE:
            return random.randint(0, OUTPUT_ACTIONS - 1)
        else:
            with torch.no_grad(): 
                state_tensor = self.process_observation(observation_dict)
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values, dim=1).item() 
                return action