import numpy as np

import gymnasium as gym
import gym_env

from ppo import PPO

dataset = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)

env = gym.make("gym_env:gym_env/PriorityScheduler-v0", data=dataset, encoder_context=5, max_priority=5)
model = PPO(env, 64)
#torch.autograd.set_detect_anomaly(True)
model.learn(10000)