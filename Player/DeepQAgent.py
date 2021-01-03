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
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
UPDATE_EVERY = 4  # how often to update the network
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

        self.policy_net.eval()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayMemory(10000)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.q = {}

    def check_state(self, s):
        pass

    def update(self, s0, s1, a, r, t):
        self.memory.push(s0, a, s1, r, t)
        self.q[s1] = self.get_q_values(s1)
        self.learn()
        pass

    def get_action(self, observation):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.policy_net.eval()

        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(observation).float().unsqueeze(0).to(device)
                action_values = self.policy_net(state)
                action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.choice(np.arange(self.action_size))

        return action

    def get_q_values(self, observation):
        with torch.no_grad():
            state = torch.tensor(observation).float().unsqueeze(0).to(device)
            action_values = self.policy_net(state)
            q_values = action_values.detach().cpu().numpy()
        return q_values

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

        state_batch = torch.tensor(batch.state).float().reshape(BATCH_SIZE, 1).to(device)
        action_batch = torch.tensor(batch.action).long().reshape(BATCH_SIZE, 1).to(device)
        next_state_batch = torch.tensor(batch.next_state).float().reshape(BATCH_SIZE, 1).to(device)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(device)
        terminal_batch = torch.tensor(batch.terminal).to(device)

        criterion = nn.MSELoss()

        self.policy_net.train()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).detach().max(1)[0].unsqueeze(1)

        next_state_values[terminal_batch] = 0.0

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = criterion(state_action_values, expected_state_action_values).to(device)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
