import os
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent


def run_experiment(experiment_id, hyperparams):
    """Run a single experiment with given hyperparameters"""
    env_name = "Door"
    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="JOINT_VELOCITY"
        ),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(env)

    agent = Agent(
        alpha=hyperparams["alpha"],
        beta=hyperparams["beta"],
        input_dims=env.observation_space.shape,
        tau=hyperparams["tau"],
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=hyperparams["layer1_size"],
        layer2_size=hyperparams["layer2_size"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        noise=hyperparams["noise"],
        update_actor_interval=hyperparams["update_actor_interval"],
        warmup=hyperparams["warmup"],
    )

    writer = SummaryWriter(f"logs/experiment_{experiment_id}")
    n_games = 1000  # Number of games per experiment

    best_score = 0
    episode_identifier = f"experiment_{experiment_id} - " + " - ".join(
        [f"{k}: {v}" for k, v in hyperparams.items()]
    )

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if not i % 10:
            agent.save_models()
            print(f"Experiment {experiment_id} - Models saved at episode {i}")

        print(f"Experiment {experiment_id} - Episode: {i} Score: {score}")


def main():
    # Define hyperparameter search space
    hyperparameter_space = {
        "alpha": [0.001, 0.0005, 0.0001],  # Actor learning rate
        "beta": [0.001, 0.0005, 0.0001],  # Critic learning rate
        "tau": [0.005, 0.001, 0.0005],  # Target network update rate
        "layer1_size": [256, 512],  # First hidden layer size
        "layer2_size": [128, 256],  # Second hidden layer size
        "batch_size": [64, 128, 256],  # Batch size
        "gamma": [0.99, 0.95],  # Discount factor
        "noise": [0.1, 0.2],  # Exploration noise
        "update_actor_interval": [2, 4],  # Actor update interval
        "warmup": [1000, 2000],  # Warmup period
    }

    # Create directory for logs if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Generate all combinations of hyperparameters
    keys = hyperparameter_space.keys()
    values = hyperparameter_space.values()
    experiments = [dict(zip(keys, combination)) for combination in product(*values)]

    # Run experiments
    for experiment_id, hyperparams in enumerate(experiments):
        print(f"\nStarting Experiment {experiment_id}")
        print(f"Hyperparameters: {hyperparams}")
        run_experiment(experiment_id, hyperparams)


if __name__ == "__main__":
    main()
