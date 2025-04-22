import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import CriticNetwork, ActorNetwork


class Agent:
    """
    The Agent class represents a reinforcement learning (RL) agent that is implemented using the
    Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm. This is a state-of-the-art
    algorithm used for continuous action spaces.

    - The agent interacts with an environment (e.g., a virtual world or a simulation),
      learns from its experiences, and tries to improve its performance over time.

    **Key Concepts in RL this agent implements:**
    - `Actor-Critic`: A combination of two deep networks (actor for policy, critic(s) to evaluate).
    - `Experience Replay`: Stores and reuses past interactions with the environment.
    - `Soft Updates`: A mechanism for transitioning between current and target networks.
    """

    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,
        n_actions=2,
        max_size=1000000,
        layer1_size=256,
        layer2_size=128,
        batch_size=100,
        noise=0.1,
    ):
        """
        Initializes the agent with key configurations and attributes needed for learning.
        Many of these attributes are involved in the TD3 algorithm's inner workings.

        **Key Parameters Passed to this Method:**
        - Various hyperparameters like `gamma`, `tau`, learning rates, network architectures, etc.
        - Actions and state space details.

        **Instance Attributes (What this Agent "knows" about itself):**
        - Neural Networks (`actor`, `critic_1`, `critic_2`, etc.):
          Deep learning models for decision-making and evaluations.
        - Replay Memory (`memory`): Stores interaction data (state, action, reward, next state).
        - Learning Attributes:
          Batch sizes, learning frequencies, exploration details, and warm-up steps.
        - Target Networks:
          Target neural networks used for stable training.
        """

        # Gamma is a discount factor for future rewards. For example, if gamma is closer to 1,
        # the agent will prioritize long-term rewards. A value closer to 0 focuses
        # more on immediate rewards (short-sighted decisions).
        self.gamma = gamma

        # Tau is used for soft updates. Instead of copying the weights of one model to another,
        # we gradually "blend" them using tau for a more stable transition.
        self.tau = tau

        # The maximum and minimum possible values of an action that the agent can take.
        # Used to constrain the actions to a valid range in the environment.
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

        # Replay memory stores the agent's experiences (state, action, reward, next state, etc.).
        # This helps to break the correlation between consecutive experiences by randomly sampling
        # "mini-batches" from memory during training.
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        # The batch size determines how many stored experiences we sample at once
        # from memory during training to update the agent's models.
        self.batch_size = batch_size

        # learn_counter tracks how often the agent is "learning".
        # This variable is important to decide when to actually update the networks.
        self.learn_counter = 0

        # Keeps track of the number of timesteps (steps or frames in an environment) taken so far.
        self.timestep = 0

        # Warmup refers to a starting period when the agent only explores (use random actions)
        # rather than relying on its neural network's predictions. This helps gather diverse
        # experience before training starts.
        self.warmup = warmup

        # Number of possible actions
        self.n_actions = n_actions

        # Controls how often the agent updates its actor network. For example, in TD3, the actor
        # is updated less frequently than the critic(s) to ensure stability.
        self.update_actor_interval = update_actor_interval

        # Actor network outputs the agent's selected action (a decision-maker).
        # Target_actor is a copy of the actor used to stabilize training by reducing
        # the likelihood of oscillations due to sudden changes.
        self.actor = ActorNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="actor",
            learning_rate=alpha,
        )
        self.target_actor = ActorNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="target_actor",
            learning_rate=alpha,
        )

        # Two critic networks (`critic_1`, `critic_2`) estimate the value function.
        # The critics give a numerical score to the state-action pairs. TD3 uses two
        # critics to address overestimation bias that was seen in earlier algorithms.
        self.critic_1 = CriticNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="critic_1",
            learning_rate=beta,
        )
        self.critic_2 = CriticNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="critic_2",
            learning_rate=beta,
        )
        self.target_critic_1 = CriticNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="target_critic_1",
            learning_rate=beta,
        )
        self.target_critic_2 = CriticNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name="target_critic_2",
            learning_rate=beta,
        )

        # Noise is added to make the agent's actions "exploratory" for better learning (try new things
        # instead of repeating the same actions). This is used for exploration during training.
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, validation=False):
        """
            Determines and returns the action for a given observation.

            **Parameters:**
            - `observation`: The current observation from the environment, which represents
              the agent's perceived state. Typically, a vector of relevant data like positions,
              velocities, or other features.
            - `validation`: A boolean flag to indicate if the agent is in validation mode
              (i.e., evaluating without exploration). In validation mode, no random noise
              is added to the action.

            **Returns:**
            - A calculated action, which is either:
              - A random action (during warm-up, when the agent is exploring the action space), or
              - An action derived from the trained actor network, potentially with added noise
                for exploration (if not in validation mode).

            **Detailed Explanation:**
            - During the warm-up phase (`self.timestep < self.warmup`), the agent chooses a random action
              with noise applied to encourage exploration.
            - After the warm-up phase, the actor network predicts an action based on the observation.
              Noise is added to the predicted action during training to allow for exploration of
              alternative actions.
            - During validation, exploration noise is disabled, allowing the agent to act deterministically.
            - The final action is clamped to ensure it remains within valid bounds.
            """

        if self.timestep < self.warmup and validation is False:
            mu = T.tensor(
                np.random.normal(scale=self.noise, size=(self.n_actions,))
            ).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(
            self.actor.device
        )
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])

        self.timestep += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        """
        Stores the agent's experience into the replay memory for future use.

        **Parameters:**
        - `state`: The state before taking the action.
        - `action`: The action taken in this state.
        - `reward`: The reward received after taking the action.
        - `next_state`: The state the agent ended up in after taking the action.
        - `done`: A boolean value indicating whether the episode has ended (e.g., the agent finished
          the task or failed at it).

        **Detailed Workflow:**
        - Experiences (state, action, reward, next_state, done) are added to the replay memory.
        - These experiences will be randomly retrieved later to train the agent's models.
        """

        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """
        This method performs the learning step: updating the actor and critic networks.

        Key Steps for Learning:
        1. Retrieve (sample) a batch of experiences from the replay memory.
        2. Compute the target Q-value (desired value) using the target networks.
        3. Calculate the "loss" (error) between predicted and desired values for critics.
        4. Backpropagate through the critic networks and update their parameters.
        5. Every few steps, update the actor and target networks.

        **Why Two Critics?**
        - To address overestimation of value functions in reinforcement learning.
        - Two critics help provide more accurate value estimates by considering the minimum
          estimate from both critics.
        """

        # Skip learning if not enough samples are in memory
        if self.memory.mem_ctr < self.batch_size * 10:
            return

        # Sample a batch of transitions from memory
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        # Convert samples to tensors for training the neural networks
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Target action calculation with noise for TD3's target smoothing
        target_actions = self.target_actor.forward(next_state)
        target_actions = target_actions + T.clamp(
            T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5
        )
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        # Compute target Q-values using the target critics
        next_q1 = self.target_critic_1.forward(next_state, target_actions)
        next_q2 = self.target_critic_2.forward(next_state, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # Set Q-values to 0 if the episode has finished (no future rewards)
        next_q1[done] = 0.0  # If done, there will be no future rewards
        next_q2[done] = 0.0  # Same for the second critic

        # Reshape the Q-values to ensure compatibility with calculations
        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        # TD3-specific: Use the minimum of the two Q-values to prevent overestimation
        next_critical_value = T.min(next_q1, next_q2)

        # Compute the target values (reward + discounted future value)
        target = reward + self.gamma * next_critical_value
        target = target.view(
            self.batch_size, 1
        )  # Reshaped to match the critic output shape

        # Zero out the gradients for both critic networks before backpropagation
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # Compute the loss between the target and the current Q-values (Mean Squared Error)
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        # Combine the two losses (total critic loss)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()  # Perform backpropagation to calculate gradients

        # Update the critic networks' parameters using their optimizers
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Increment the learning counter to track how many updates have happened
        self.learn_counter += 1

        # TD3-specific: Only update the actor network every few learning steps
        if self.learn_counter % self.update_actor_interval != 0:
            return

        # Zero out the gradients for the actor network before backpropagation
        self.actor.optimizer.zero_grad()

        # Compute the actor loss using the first critic's evaluation
        # The actor tries to maximize the critic's estimated Q-value of its actions
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(
            actor_q1_loss
        )  # Negative because we want to maximize Q-value
        actor_loss.backward()  # Perform backpropagation to compute gradients for the actor

        # Update the actor network's parameters using its optimizer
        self.actor.optimizer.step()

        # Update the target networks with soft updates (controlled by tau)
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """
        Updates the weights of the target networks using soft updates. Soft updates ensure
        that changes to the networks are gradual and do not destabilize training.

        **Parameters:**
        - `tau`: A factor determining how much of the current network's parameters
          is blended into the target network.

        Soft Update Formula (for each weight `w`):
        target_weight = tau * current_weight + (1 - tau) * target_weight
        """

        if tau is None:  # Use the default tau value if none provided
            tau = self.tau

        # Retrieve and prepare all network parameters as dictionaries
        actor_params = self.actor.named_parameters()
        critic1_params = self.critic_1.named_parameters()
        critic2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic1_params = self.target_critic_1.named_parameters()
        target_critic2_params = self.target_critic_2.named_parameters()

        # Convert parameters into dictionaries for easy access
        critic1_state_dict = dict(critic1_params)
        critic2_state_dict = dict(critic2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic1_state_dict = dict(target_critic1_params)
        target_critic2_state_dict = dict(target_critic2_params)

        # SOFT UPDATE: For all parameters in each critic and actor, do a weighted update
        for name in critic1_state_dict:
            critic1_state_dict[name] = (
                tau * critic1_state_dict[name].clone()
                + (1 - tau) * target_critic1_state_dict[name].clone()
            )

        for name in critic2_state_dict:
            critic2_state_dict[name] = (
                tau * critic2_state_dict[name].clone()
                + (1 - tau) * target_critic2_state_dict[name].clone()
            )

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        # Load the updated parameters back into the target networks
        self.target_critic_1.load_state_dict(critic1_state_dict)
        self.target_critic_2.load_state_dict(critic2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print("Successfully loaded models")
        except:
            print("Failed to load models. Starting from scratch")
