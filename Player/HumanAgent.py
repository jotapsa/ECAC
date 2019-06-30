from Player.Api.Agent import *


class HumanAgent(Agent):

    def __init__(self, action_map=None):
        self.action_map = action_map if action_map is not None else {'w': 3, 'a': 1, 's': 2, 'd': 0}

    def check_state(self, s):
        pass

    def update(self, s0, s1, a, r, t):
        pass

    def get_action(self, s):
        return self.action_map[input("Action: ")]
