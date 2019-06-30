from Util.Projection import *
from Player.Api.Agent import *


class MixedAgent(Agent):

    def __init__(self, pi):
        self.pi = pi  # policy (distribution)
        self.n_actions = len(next(iter(pi.values())))

    def check_state(self, s):
        """

        :param s: state
        """
        if s not in self.pi:
            self.pi[s] = np.full(self.n_actions, 1 / self.n_actions)

    def update(self, s0, s1, a, r, t):
        pass

    def get_action(self, s):
        """

        :param s: state
        :return: action
        """
        self.check_state(s)
        return np.random.choice(self.n_actions, p=self.pi[s])

    def __str__(self):
        return '{' + ', '.join([str(e) for e in self.pi]) + '}'
