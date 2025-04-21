# Standard library import for handling file paths
import os

# PyTorch imports with common aliases
import torch as T  # Main PyTorch library
import torch.nn.functional as F  # Contains activation functions and other functional operations
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.distributions.normal import Normal  # For handling normal distributions
import numpy as np  # For numerical operations


class CriticNetwork(nn.Module):
    # This class inherits from PyTorch's base neural network class

    def __init__(
        self,
        input_dims,  # Size of the input (state space)
        n_actions,  # Number of possible actions
        fc1_dims=256,  # Size of first fully connected layer
        fc2_dims=128,  # Size of second fully connected layer
        name="critic",  # Name of the network
        checkpoint_dir="tmp/td3",  # Directory to save model checkpoints
        learning_rate=10e-3,  # Learning rate for optimization
    ):
        super(CriticNetwork, self).__init__()  # Initialize the parent class

        # Store all parameters as instance variables
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_td3")

        # Define the neural network layers
        # First layer combines state and action as input
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        # Second hidden layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Output layer produces a single Q-value
        self.q1 = nn.Linear(self.fc2_dims, 1)

        # Setup the optimizer (AdamW with weight decay for regularization)
        self.optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=0.005
        )

        # Set up the device (GPU if available, otherwise CPU)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        print(f"Created Critic Network on device: {self.device}")

        # Move the network to the appropriate device
        self.to(self.device)

    def forward(self, state, action):
        # Combine state and action into one tensor
        action_value = self.fc1(T.cat([state, action], dim=1))
        # Apply ReLU activation
        action_value = F.relu(action_value)
        # Second layer
        action_value = self.fc2(action_value)
        # Another ReLU activation
        action_value = F.relu(action_value)
        # Final layer to get Q-value
        q1 = self.q1(action_value)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_state(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    # Similar initialization but simpler network structure
    # Actor network takes only the state as input and outputs actions

    def __init__(
        self,
        input_dims,
        n_actions=2,
        fc1_dims=256,
        fc2_dims=128,
        name="actor",
        checkpoint_dir="tmp/td3",
        learning_rate=10e-3,
    ):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_td3")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        print(f"Created Actor Network on device: {self.device}")

        self.to(self.device)

    def forward(self, state):
        # First layer
        x = self.fc1(state)
        x = F.relu(x)
        # Second layer
        x = self.fc2(x)
        x = F.relu(x)
        # Output layer with tanh activation to bound the actions
        x = T.tanh(self.output(x))
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
