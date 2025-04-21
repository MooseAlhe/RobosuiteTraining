import numpy as np  # Importing the NumPy library, which is a powerful library for numerical computations in Python.


# It is commonly used for working with arrays, mathematical operations, and data manipulation.


# The ReplayBuffer class is designed to store and manage experiences (transitions) for reinforcement learning.
# "Transitions" in this case include the current state, action taken, reward received, next state encountered,
# and whether the episode terminated ("done" status). This storage is crucial for training agents in reinforcement learning.
class ReplayBuffer:

    def __init__(self, max_size, input_shape, n_actions):
        """
        Initializes the replay buffer storage.

        Arguments:
        - max_size : int : The maximum number of transitions the buffer can store.
        - input_shape : tuple : The shape of the state, typically representing how the environment provides data.
        - n_actions : int : The number of actions available in the environment.

        Creates and initializes the memory buffers for different components of a transition.
        """
        self.mem_size = max_size  # Sets the total capacity (maximum number of transitions) for the buffer.
        self.mem_ctr = 0  # Counter to track how many transitions have been added to the buffer so far.

        # Allocate an array to store states, with size `max_size` for the number of transitions,
        # and `input_shape` describing the dimensionality of each state.
        self.state_memory = np.zeros((self.mem_size, *input_shape))

        # Similar to `state_memory`, but stores the "next states" encountered after every action.
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))

        # Stores the actions taken at each transition. The array size is `max_size x n_actions`, where
        # each row is a one-hot or numerical representation of the action.
        self.action_memory = np.zeros((self.mem_size, n_actions))

        # Stores the rewards corresponding to each transition. Rewards are scalar values for each step.
        self.reward_memory = np.zeros(self.mem_size)

        # Stores whether the episode terminated after each transition ("done" flag).
        # A boolean flag is used to indicate the end of an episode.
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a single transition (state, action, reward, next state, done) in the replay buffer.
        Older transitions are overwritten when the buffer reaches capacity (`max_size`).

        Arguments:
        - state : array : The current state of the environment.
        - action : array : The action taken by the agent.
        - reward : float : The reward obtained after taking the action.
        - next_state : array : The state after the action was taken.
        - done : bool : Whether the episode ended after the action.
        """
        # Calculate the index where this transition will be stored.
        # This ensures a circular buffer implementation, where newer data overwrites the oldest data once the buffer is full.
        index = self.mem_ctr % self.mem_size

        # Store the components of the transition in their respective memory arrays.
        self.state_memory[index] = (
            state  # Store the current state at the computed index.
        )
        self.next_state_memory[index] = next_state  # Store the next state.
        self.action_memory[index] = action  # Store the action taken.
        self.reward_memory[index] = reward  # Store the reward received.
        self.terminal_memory[index] = done  # Store whether the episode ended.

        # Increment the counter to track the number of transitions stored.
        # This counter continues to grow beyond `max_size`, but the modulo operation ensures overwriting.
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        """
        Samples a random batch of transitions from the buffer for training.

        Arguments:
        - batch_size : int : The number of transitions to sample.

        Returns:
        - states : array : Batch of sampled states.
        - actions : array : Batch of sampled actions.
        - rewards : array : Batch of sampled rewards.
        - next_states : array : Batch of sampled next states.
        - dones : array : Batch of sampled "done" flags.
        """
        # Get the number of valid transitions in the buffer. This is the minimum of
        # `mem_ctr` (total transitions added) and `mem_size` (buffer's maximum capacity).
        max_mem = min(self.mem_ctr, self.mem_size)

        # Randomly select `batch_size` indices from the range `[0, max_mem)`. This ensures
        # that the sampled transitions are valid and within the populated portion of the buffer.
        batch = np.random.choice(max_mem, batch_size)

        # Use the randomly selected indices to sample transitions from the buffer.
        states = self.state_memory[batch]  # Retrieve the batch of states.
        next_states = self.next_state_memory[
            batch
        ]  # Retrieve the batch of next states.
        actions = self.action_memory[batch]  # Retrieve the batch of actions.
        rewards = self.reward_memory[batch]  # Retrieve the batch of rewards.
        dones = self.terminal_memory[batch]  # Retrieve the batch of "done" flags.

        # Return the sampled transitions as a tuple. This tuple contains all the components needed for training the agent.
        return states, actions, rewards, next_states, dones
