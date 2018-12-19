import numpy as np
import random
from collections import namedtuple, deque

from model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, state_size, h_size, seed):
        """Initialize an Agent object."""

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.network_local = Network(action_size, state_size, h_size, seed).to(device)
        self.network_target = Network(action_size, state_size, h_size, seed).to(device)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, observation, action, reward, next_observation, done):
        # Save experience in replay memory
        # state = self.network_local.state_representation(observation)
        # next_state = self.network_local.state_representation(next_observation)

        self.memory.add(observation, action, reward, next_observation, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, observation, eps=0.):
        """Returns actions for given state as per current policy."""
        
        observation = torch.from_numpy(observation).float().permute(
            2, 0, 1).unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).to(device)

        self.network_local.eval()
        with torch.no_grad():
            action_values, _ = self.network_local(observation)
        self.network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values[0].cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""

        observation, actions, rewards, next_observation, dones = experiences
        
        # observation = torch.from_numpy(observation).float().permute(2, 0, 1).unsqueeze(0).to(device)
        # next_observation = torch.from_numpy(next_observation).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Q-Learning
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.network_target(next_observation)

        Q_targets_next.detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected, next_state_predicted = self.network_local(observation)
        Q_expected.gather(1, actions)

        # Compute Q loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        
        # Self Supervision
        # TODO
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().permute(0, 3, 1, 2).to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().permute(0, 3, 1, 2).to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
