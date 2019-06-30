from Util.Projection import *
from Player.QLAgent import *
import numpy as np


class ABWPL(QLAgent):
    class StateStatistics:
        def __init__(self):
            self.adjusted_boundary = 0
            self.prev_best_action = 0
            self.steps_since_update = 0
            self.avg_update_steps = 1

    def __init__(self, l_rate, y, e_rate, n_actions, pi_l_rate, min_e_rate):
        """

        :param l_rate: learning rate
        :param y: future rewards importance
        :param e_rate: exploration_rate
        :param n_actions: number of actions
        :param pi_l_rate: policy learning rate
        :param min_e_rate:
        """
        super().__init__(l_rate, y, e_rate, n_actions)
        self.pi_l_rate = pi_l_rate
        self.pi = {}  # policy (distribution)
        self.ss = {}  # state statistics
        self.min_e_rate = min_e_rate

    def check_state(self, s):
        """

        :param s: state
        """
        super().check_state(s)
        if s not in self.pi:
            self.pi[s] = np.full(self.n_actions, 1 / self.n_actions)
            self.ss[s] = ABWPL.StateStatistics()

    def update(self, s0, s1, a, r, t):
        """

        :param s0: previous state
        :param s1: current state
        :param a: action
        :param r: reward
        :param t: terminal
        """
        super().update(s0, s1, a, r, t)
        best_action = np.argmax(self.q[s0])
        self.ss[s0].steps_since_update += 1
        # if best action has not changed
        if best_action == self.ss[s0].prev_best_action:
            # and we've stepped over our avg update steps
            if self.ss[s0].steps_since_update > self.ss[s0].avg_update_steps:
                # we start adjusting our boundary
                self.ss[s0].adjusted_boundary += 0.5 / self.ss[s0].avg_update_steps
                # but without going over the limit
                if self.ss[s0].adjusted_boundary >= 0.5:
                    self.ss[s0].adjusted_boundary = 0.5 - self.pi_l_rate
        # otherwise if best action is changed
        else:
            # we reset our adjusted boundary
            self.ss[s0].adjusted_boundary = 0
            # and update our avg_update_steps IF they're over "avg update steps / 2" (to ignore noise)
            if self.ss[s0].steps_since_update > self.ss[s0].avg_update_steps / 2:
                # moving average of last 2 windows
                self.ss[s0].avg_update_steps = (self.ss[s0].avg_update_steps + self.ss[s0].steps_since_update) / 2
            # steps since last change
            self.ss[s0].steps_since_update = 0
        self.ss[s0].prev_best_action = best_action
        for action in range(self.n_actions):
            difference = 0
            # compute difference between this reward and average reward
            for action2 in range(self.n_actions):
                difference += self.q[s0][action] - self.q[s0][action2]
            difference /= self.n_actions - 1
            # scale to sort of normalize the effect of a policy
            if difference > 0:
                # when we are favoring the best action, we take a lower delta_policy, so that we move slowly
                delta_policy = 1 - self.pi[s0][action]
            else:
                # when we are favoring the worst action, we take a higher delta_policy, so that we move quickly
                delta_policy = self.pi[s0][action]
            delta_policy = delta_policy * (1 - (self.ss[s0].adjusted_boundary * 2)) + self.ss[s0].adjusted_boundary
            self.pi[s0][action] += self.pi_l_rate * difference * delta_policy
        # project policy back into valid policy space
        self.pi[s0] = projection(self.pi[s0], self.min_e_rate)

    def get_action(self, s):
        """

        :param s: state
        :return: action
        """
        # choose random action with probability e_rate
        if random() < self.e_rate:
            return int(random() * self.n_actions)
        self.check_state(s)
        return np.random.choice(self.n_actions, p=self.pi[s])

    def __str__(self):
        return '[' + ', '.join([str(elem) for elem in self.q]) + ']'
