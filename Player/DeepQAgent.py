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
        self.target_net = DeepQNetwork(
            lr=0.003,
            input_dims=1,
            fc1_neurons=128,
            fc2_neurons=128,
            n_actions=n_actions
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def check_state(self, s):
        pass

    def update(self, s0, s1, a, r, t):
        self.memory.push([float(s0)], a, [float(s1)], r, t)
        self.learn()
        pass

    def get_action(self, observation):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1
        if sample > eps_threshold:
            state = torch.tensor([float(observation)]).to(self.policy_net.device)
            actions = self.policy_net.forward(state)
            action = actions.max().item()
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

        batch_index = np.arange(BATCH_SIZE, dtype=np.int64)

        state_batch = torch.tensor(batch.state).to(self.policy_net.device)
        action_batch = torch.tensor(batch.action).to(self.policy_net.device)
        next_state_batch = torch.tensor(batch.next_state).to(self.policy_net.device)
        reward_batch = torch.tensor(batch.reward).to(self.policy_net.device)
        terminal_batch = torch.tensor(batch.terminal).to(self.policy_net.device)

        state_action_values = self.policy_net.forward(state_batch)
        next_state_values = self.target_net.forward(next_state_batch)
        next_state_values[terminal_batch] = 0.0

        reward_batch = reward_batch.reshape(128, 1)
        reward_batch = reward_batch.expand(-1, 4)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # # Compute loss
        loss = self.policy_net.loss(state_action_values, expected_state_action_values ).to(self.policy_net.device)
        # Optimize the model
        self.policy_net.optimizer.zero_grad()
        # loss.backward()
        self.policy_net.optimizer.step()
