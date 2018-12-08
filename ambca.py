import numpy as np
from collections import deque

import gym
import torch

from dqn_agent import Agent


env = gym.make('CarRacing-v0')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


agent = Agent()


episodes = 100
steps = 150

for i_episode in range(episodes):
    obs = env.reset()
    score = 0
    for t in range(steps):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward
        if done:
            break
