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

BATCH_SIZE = 64
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
UPDATE_EVERY = 4        # how often to update the network
TARGET_UPDATE = 10


# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, state_size, action_size, fc1_nodes=64, fc2_nodes=64):
        # Layers
        super().__init__()
        self.f1 = nn.Linear(state_size, fc1_nodes)
        self.f2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.f3 = nn.Linear(fc2_nodes, action_size)

        self.optimizer = optim.RMSprop(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()

    def forward(self, state):
        output = F.relu(self.f1(state))
        output = F.relu(self.f2(output))
        actions = self.f3(output)
        return actions


class DeepQAgent(Agent):
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.steps_done = 0

        self.policy_net = DeepQNetwork(
            state_size=state_size,
            action_size=action_size
        ).to(device)
        self.target_net = DeepQNetwork(
            state_size=state_size,
            action_size=action_size
        ).to(device)

        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()

        # Replay memory
        self.memory = ReplayMemory(10000)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def check_state(self, s):
        pass

    def update(self, s0, s1, a, r, t):
        # self.memory.push([float(s0)], a, [float(s1)], r, t)
        self.memory.push(s0, a, s1, r, t)
        self.learn()
        pass

    def get_action(self, observation):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        # self.qnetwork_local.eval()
        # with torch.no_grad():
        #     action_values = self.qnetwork_local(state)
        # # self.qnetwork_local.train()

        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(observation).float().unsqueeze(0).to(device)
                action_values = self.policy_net(state)
                action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.choice(np.arange(self.action_size))

        return action

    def learn(self):
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step != 0:
            return

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        batch_index = np.arange(BATCH_SIZE, dtype=np.int64)

        state_batch = torch.tensor(batch.state).float().reshape(BATCH_SIZE, 1).to(device)
        action_batch = torch.tensor(batch.action).to(device)
        next_state_batch = torch.tensor(batch.next_state).float().reshape(BATCH_SIZE, 1).to(device)
        reward_batch = torch.tensor(batch.reward).to(device)
        terminal_batch = torch.tensor(batch.terminal).to(device)

        state_action_values = self.policy_net.forward(state_batch)
        next_state_values = self.target_net.forward(next_state_batch)
        next_state_values[terminal_batch] = 0.0

        reward_batch = reward_batch.reshape(BATCH_SIZE, 1)
        reward_batch = reward_batch.expand(-1, self.action_size)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # # Compute loss
        loss = self.policy_net.loss(state_action_values, expected_state_action_values).to(device)
        # Optimize the model
        self.policy_net.optimizer.zero_grad()
        # loss.backward()
        self.policy_net.optimizer.step()
