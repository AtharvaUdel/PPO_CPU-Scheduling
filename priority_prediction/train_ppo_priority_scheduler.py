import numpy as np
import torch

import gymnasium as gym
import gym_env

from ppo import PPO

dataset1 = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
dataset2 = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
dataset3 = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
dataset4 = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
dataset5 = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)

env = gym.make("gym_env:gym_env/PriorityScheduler-v0", data=dataset1, encoder_context=10, max_priority=10)
model = PPO(env, 64)

n_steps = 5

model.learn(n_steps)
env.reset(options={'new_data': dataset2})
model.learn(n_steps)
env.reset(options={'new_data': dataset3})
model.learn(n_steps)
env.reset(options={'new_data': dataset4})
model.learn(n_steps)
env.reset(options={'new_data': dataset5})
model.learn(n_steps)

torch.save(model.actor.state_dict(), 'model_weights/ml_priority_scheduler.pt')
