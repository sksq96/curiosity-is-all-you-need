import numpy as np
from collections import deque

import gym
import torch

from dqn_agent import Agent
from utils import resize


SEED = 0

env = gym.make('Breakout-v0')
# env = gym.make('CarRacing-v0')
env.seed(SEED)

obs_size, action_size = env.observation_space.shape, env.action_space.n
print('State shape: ', obs_size)
print('Number of actions: ', action_size)

state_size = 1024
h_size = 128
agent = Agent(action_size, state_size, h_size=h_size, seed=SEED)


episodes = 100
steps = 150

for i_episode in range(episodes):
    obs = env.reset()
    obs = resize(obs, size=64)
    score = 0
    for t in range(steps):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        print(reward)
        next_obs = resize(next_obs, size=64)
        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward
        if done:
            break
