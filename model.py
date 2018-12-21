import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, state_size, h_size, seed, image_channels=3, device=None):
        """Initialize parameters and build model."""
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.state_representation = StateRepesentation(image_channels)
        self.temporal_representation = HistoryRepresentation(state_size, h_size)
        self.hidden = self.temporal_representation.init_hidden()

        self.forward_prediction = nn.Sequential(
            nn.Linear(h_size, state_size)
        )

        self.q_value_prediction = nn.Sequential(
            nn.Linear(h_size, action_size)
        )

    def forward(self, observation):
        """Build a network that maps observation -> (action values, next state)"""

        state = self.state_representation(observation)
        self.hidden = self.temporal_representation(
            state, self.temporal_representation.init_hidden())

        h, _ =  self.hidden
        return self.q_value_prediction(h), self.forward_prediction(h)


class HistoryRepresentation(nn.Module):
    def __init__(self, in_size, h_size):
        super(HistoryRepresentation, self).__init__()
        self.h_size = h_size
        self.rnn = nn.LSTMCell(in_size, h_size)

    def forward(self, x, h):
        h, c = self.rnn(x, h)
        return (h, c)

    def init_hidden(self, bsz=32, device=None):
        return (torch.zeros(bsz, self.h_size).to(device),
                torch.zeros(bsz, self.h_size).to(device))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class StateRepesentation(nn.Module):
    def __init__(self, image_channels=3):
        super(StateRepesentation, self).__init__()
        # (3, 64, 64) --> 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            Flatten()
        )

    def forward(self, x):
        return self.encoder(x)

