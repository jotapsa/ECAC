from Player.Api.Agent import Agent
from random import random
import numpy as np


class QLAgent(Agent):

    def __init__(self, l_rate, y, e_rate, n_actions):
        """

        :param l_rate: learning rate
        :param y: future rewards importance
        :param e_rate: exploration_rate
        :param n_actions: number of actions
        """
        self.l_rate = l_rate
        self.y = y
        self.e_rate = e_rate
        self.n_actions = n_actions
        self.q = {}  # Q values

    def check_state(self, s):
        """

        :param s: state
        """
        if s not in self.q:
            self.q[s] = np.zeros(self.n_actions)

    def update(self, s0, s1, a, r, t):
        """

        :param s0: previous state
        :param s1: current state
        :param a: action
        :param r: reward
        :param t: terminal
        """
        # register states if they don't exist already
        self.check_state(s0)
        self.check_state(s1)
        # update the Q-table of agent
        self.q[s0][a] = (1 - self.l_rate) * self.q[s0][a] + self.l_rate * (r + (1 - t) * self.y * max(self.q[s1]))

    def get_action(self, s):
        """

        :param s: state
        :return: action
        """
        # choose random action with probability e_rate
        if random() < self.e_rate:
            return int(random() * self.n_actions)
        self.check_state(s)
        return np.argmax(self.q[s])

    def __str__(self):
        return '{' + ', '.join([str(elem) for elem in self.q]) + '}'
