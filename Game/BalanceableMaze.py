import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque

DEFAULT_LENGTH = 7
DEFAULT_WIDTH = 7
DEFAULT_WALL = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
                (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),
                (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),
                (2, 2), (3, 2), (4, 2),
                (2, 4), (3, 4), (4, 4)}
DEFAULT_POS0 = (1, 5)
DEFAULT_GM_POS0 = (2, 2)
DEFAULT_GOAL = (5, 1)

# RGB colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


class BalanceableMaze(gym.Env):

    def __init__(self, length=DEFAULT_LENGTH, width=DEFAULT_WIDTH, goal=DEFAULT_GOAL, pos0=DEFAULT_POS0,
                 gm_pos0=DEFAULT_GM_POS0, wall=None,
                 scale=50):
        self.length = length
        self.width = width
        self.n_states = self.length * self.width
        self.goal = goal
        self.pos0 = pos0
        self.gm_pos0 = gm_pos0
        self.wall = DEFAULT_WALL if wall is None else wall

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(1)

        self.pos = [self.pos0[0], self.pos0[1]]
        self.gm_pos = [self.gm_pos0[0], self.gm_pos0[1]]

        # render
        self.scale = scale
        pygame.init()

        self.steps = 0

        # meta
        self.win_history = deque([0])
        self.win_rate = 0.

        # balance state
        self.locked = False
        self.locked_state = 0

    def step(self, actions):

        self.steps += 1

        # save old pos
        old_pos = (self.pos[0], self.pos[1])

        # move
        action = actions[0]
        if action == 0:
            self.pos[0] = (self.pos[0] + 1) % self.width
        elif action == 1:
            self.pos[0] = (self.pos[0] - 1) % self.width
        elif action == 2:
            self.pos[1] = (self.pos[1] + 1) % self.length
        elif action == 3:
            self.pos[1] = (self.pos[1] - 1) % self.length

        # collision
        collision = (self.pos[0], self.pos[1]) in self.wall or (
                    self.pos[0] == self.gm_pos[0] and self.pos[1] == self.gm_pos[1])
        if collision:
            self.pos = [old_pos[0], old_pos[1]]

        # terminal state
        terminal = self.pos[0] == self.goal[0] and self.pos[1] == self.goal[1]
        if terminal:
            self.win_history.appendleft(1)
        elif self.steps >= 25:
            self.win_history.appendleft(0)
        if len(self.win_history) > 10:
            self.win_history.pop()
        self.win_rate = np.average(self.win_history)

        # get reward
        if terminal:
            reward = [1.]
        elif collision:
            reward = [-.5]
        else:
            reward = [0.]

        # get state
        state = self.__state(self.pos)

        # game master
        balance_reward = [0.]
        balance_state = self.__balance_state()

        if len(actions) > 1:

            # save old pos
            gm_old_pos = (self.gm_pos[0], self.gm_pos[1])

            # move
            gm_action = actions[1]
            if gm_action == 0:
                self.gm_pos[0] = (self.gm_pos[0] + 1) % self.width
            elif gm_action == 1:
                self.gm_pos[0] = (self.gm_pos[0] - 1) % self.width
            elif gm_action == 2:
                self.gm_pos[1] = (self.gm_pos[1] + 1) % self.length
            elif gm_action == 3:
                self.gm_pos[1] = (self.gm_pos[1] - 1) % self.length

            # collision
            gm_collision_wall = (self.gm_pos[0], self.gm_pos[1]) in self.wall
            gm_collision_player = self.pos[0] == self.gm_pos[0] and self.pos[1] == self.gm_pos[1]
            if gm_collision_wall or gm_collision_player:
                self.gm_pos = [gm_old_pos[0], gm_old_pos[1]]

            # get balance reward
            if terminal:
                if balance_state == 1:
                    balance_reward = [-1.]
            elif balance_state == 0:
                if self.gm_pos[0] == self.gm_pos0[0] and self.gm_pos[1] == self.gm_pos0[1]:
                    balance_reward = [1.]
            elif gm_collision_wall:
                balance_reward = [-.5]

        # get joint_state
        # joint_state = self.__joint_state(state, self.__state(self.gm_pos))

        # get gm state
        gm_state = self.__gm_state(self.gm_pos, balance_state)

        # get gm joint_state
        gm_joint_state = self.__joint_state(state, gm_state)

        # return step results
        return [state, gm_joint_state], reward + balance_reward, terminal, None

    def reset(self):

        self.steps = 0

        # set pos to pos0
        self.pos = [self.pos0[0], self.pos0[1]]

        # set gm_pos to gm_pos0
        self.gm_pos = [self.gm_pos0[0], self.gm_pos0[1]]

        # get state
        state = self.__state(self.pos)

        # get gm state
        gm_state = self.__gm_state(self.gm_pos, self.__balance_state())

        # get joint_state
        # joint_state = self.__joint_state(state, self.__state(self.gm_pos))

        # get gm joint_state
        gm_joint_state = self.__joint_state(state, gm_state)

        # return step results
        return [state, gm_joint_state]

    def render(self, mode='human'):

        # set the height and width of the screen
        size = [self.scale * self.length, self.scale * self.width]
        screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Maze")

        # clear the screen and set the screen background
        balance_state = self.__balance_state()
        if balance_state == 1:
            screen.fill(YELLOW)
        else:
            screen.fill(WHITE)

        # draw goal
        pygame.draw.rect(screen, GREEN, [self.goal[0] * self.scale, self.goal[1] * self.scale, self.scale, self.scale])

        # draw walls
        for wall in self.wall:
            pygame.draw.rect(screen, BLACK, [wall[0] * 50, wall[1] * 50, 50, 50])

        # draw player
        pygame.draw.rect(screen, BLUE, [self.pos[0] * self.scale, self.pos[1] * self.scale, self.scale, self.scale])

        # draw game master
        pygame.draw.rect(screen, RED,
                         [self.gm_pos[0] * self.scale, self.gm_pos[1] * self.scale, self.scale, self.scale])

        pygame.display.flip()

    def __state(self, v):
        return v[0] + self.length * v[1]

    def __gm_state(self, v, balance_state):
        return self.__state(v) + self.n_states * balance_state

    def __joint_state(self, s, gm_s):
        return s + self.n_states * gm_s

    def __balance_state(self):
        if self.locked:
            return self.locked_state
        return 0 if self.win_rate <= .5 else 1

    def lock_balance_state(self, state):
        self.locked = True
        self.locked_state = state

    def free_balance_state(self):
        self.locked = False
