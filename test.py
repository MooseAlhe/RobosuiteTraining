import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

if __name__ == "__main__":

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="JOINT_VELOCITY"
        ),
        has_renderer=True,
        use_camera_obs=False,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    alpha = 0.001
    beta = 0.001
    batch_size = 120
    layer1_size = 256
    layer2_size = 128

    agent = Agent(
        alpha=alpha,
        beta=beta,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
    )

    n_games = 3
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate: {alpha} - critic_learning_rate: {beta} - layer1_size: {layer1_size} - layer2_size: {layer2_size}"

    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = next_observation
            time.sleep(0.03)

        print(f"Episode: {i} Score: {score}")
