# TIL-25 Data Chefs - Reinforcement Learning Challenge

**Hackathon:** TIL-25 Hackathon
**Team:** Data Chefs
**Author:** lolkabash

## 📖 Description

This repository contains the solution for the Reinforcement Learning (RL) challenge as part of the TIL-25 Hackathon. The project focuses on training an RL agent to solve a specific task or environment. All development and experimentation were primarily conducted within Jupyter Notebooks.

## 💻 Technologies Used

*   **Jupyter Notebook:** Primary environment for RL agent development, training, and visualization.
*   **Python:** Core programming language.
*   **RL libraries/frameworks used: OpenAI Gym, Stable Baselines3, RLlib, PyTorch, TensorFlow.)**
*   **Key Python libraries: NumPy, Pandas, Matplotlib.**

## ⚙️ Working Process & Solution

This section outlines the general steps taken to address the RL challenge.

### 1. Environment Definition & Understanding
*   **Environment Used:** (Describe the RL environment, e.g., a classic control problem from OpenAI Gym, a custom-built environment. Specify observation space, action space.)
*   **Problem Formulation:** (Clearly define the goal of the RL agent.)

### 2. Agent & Algorithm Selection
*   **Algorithm Choice:** (Explain why a particular RL algorithm was chosen, e.g., Q-Learning, DQN, PPO, A2C, SAC. Justify based on the environment's characteristics - discrete/continuous action space, model-free/model-based.)
*   **Agent Architecture:** (If a neural network was used for function approximation, describe its architecture - layers, activation functions.)

### 3. Training Process
*   **Environment Setup:** (Briefly mention the setup for running the notebooks and training.)
*   **Hyperparameter Tuning:** (Key hyperparameters for the chosen algorithm, learning rate, discount factor (gamma), exploration strategy (e.g., epsilon-greedy), buffer size, batch size, update frequency.)
*   **Reward Shaping:** (If applicable, describe any reward shaping techniques used to guide the agent.)
*   **Training Iterations & Convergence:** (How long was the agent trained? Any observations about convergence or learning stability?)
*   **Challenges Faced:** (Any significant challenges during training, e.g., sparse rewards, unstable learning, and how they were addressed.)

### 4. Evaluation
*   **Metrics Used:** (How was the agent's performance measured? E.g., cumulative reward, episode length, success rate.)
*   **Evaluation Protocol:** (How many episodes were run for evaluation? Was it on a separate test environment or using a deterministic policy?)
*   **Performance Visualizations:** (E.g., learning curves showing reward over episodes/timesteps.)

### 5. Results & Key Findings
*   **Final Agent Performance:** (Summarize the best performance achieved by the agent.)
*   **Insights:** (Any interesting behaviors learned by the agent or insights from the training process.)
*   **(Consider adding GIFs or videos of the trained agent interacting with the environment if possible.)**

## 🚀 Setup and Usage

### Prerequisites
*   Python (version, e.g., 3.8+)
*   Jupyter Notebook/JupyterLab
*   (List other major dependencies, e.g., specific versions of RL libraries)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/lolkabash/til-25-data-chefs-RL.git
    cd til-25-data-chefs-RL
    ```
2.  Install dependencies:
    *(Provide instructions, e.g., using pip)*
    ```bash
    pip install -r requirements.txt
    ```
    *(Or if you used Conda)*
    ```bash
    # conda env create -f environment.yml
    # conda activate your_env_name
    ```

### Running the Notebooks
*   Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    # or
    # jupyter lab
    ```
*   Navigate to the main notebook(s) that demonstrate the RL agent's training and evaluation.
    *(List the key notebooks and their purpose, e.g., `RL_Agent_Training.ipynb`, `Evaluate_Agent.ipynb`)*

## 📁 File Structure (Optional - Example)

```
til-25-data-chefs-RL/
├── notebooks/                  # All Jupyter notebooks for the RL challenge
│   ├── 01_environment_setup.ipynb
│   ├── 02_agent_training.ipynb
│   └── 03_evaluation.ipynb
├── src/                        # (Optional) Any utility Python scripts or custom environment code
├── data/                       # (Optional) Data for custom environments or saved agent policies/weights
├── .gitignore
└── README.md
```
*(Adjust the file structure to match your actual repository layout.)*

## 🙏 Acknowledgements (Optional)
*   Mention any RL frameworks, environments, or research papers that inspired your work.

# RL

Your RL challenge is to direct your agent through the game map while interacting with other agents and completing challenges.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/rl` route on port `5004`. It is a JSON object structured as such:

```JSON
{
  "instances": [
    {
      "observation": {
        "viewcone": [[0, 0, ..., 0], [0, 0, ..., 0], ... , [0, 0, ..., 0]],
        "direction": 0,
        "location": [0, 0],
        "scout": 0,
        "step": 0
      }
    }
  ]
}
```

The observation is a representation of the inputs the agent senses in its environment. See the [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications) to learn how to interpret the observation.

The length of the `instances` array is 1.

During evaluation for Qualifiers, a GET request will be sent to the `/reset` route to signal that a round has ended, all agents are being reset to their starting positions (possibly with new roles), and any persistent state information your code may have stored must be cleared.

### Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        {
            "action": 0
        }
    ]
}
```

The action is an integer representing the next movement your agent intends to take. See the [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications) for a list of possible movements.
