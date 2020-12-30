import random
import math
from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from Player.Api.Agent import Agent

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(torch.nn.Module):
    def __init__(self, lr, input_dims, fc1_neurons, fc2_neurons, n_actions):
        # Layers
        super().__init__()
        self.f1 = nn.Linear(input_dims, fc1_neurons)
        self.f2 = nn.Linear(fc1_neurons, fc2_neurons)
        self.f3 = nn.Linear(fc2_neurons, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        output = F.relu(self.f1(state))
        output = F.relu(self.f2(output))
        actions = self.f3(output)
        return actions


class DeepQAgent(Agent):
    def __init__(self, n_actions):
        self.steps_done = 0
        self.action_space = [i for i in range(n_actions)]

        self.memory = ReplayMemory(10000)
        self.policy_net = DeepQNetwork(
            lr=0.003,
            input_dims=1,
            fc1_neurons=128,
            fc2_neurons=128,
            n_actions=n_actions
        )


    def check_state(self, s):
        pass

    def update(self, s0, s1, a, r, t):
        self.memory.push(s0, a, s1, r, t)
        pass

    def get_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1
        if sample > eps_threshold:
            actions = self.policy_net.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        terminal_batch = torch.cat(batch.terminal)

        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        # loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
