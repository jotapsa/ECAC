import gym
from gym import spaces
import pygame
from functools import reduce
from collections import deque
import numpy as np

DEFAULT_LENGTH = 7
DEFAULT_WIDTH = 7
DEFAULT_WALL = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
                (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),
                (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),
                (2, 2), (3, 2), (4, 2),
                (2, 4), (3, 4), (4, 4)}
DEFAULT_POS0 = [(1, 5), (5, 5)]
DEFAULT_GM_POS0 = (3, 2)
DEFAULT_GOAL = (3, 1)

# RGB colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)


class BalanceableMazeMP(gym.Env):

    def __init__(self, length=DEFAULT_LENGTH, width=DEFAULT_WIDTH, goal=DEFAULT_GOAL, pos0=None,
                 gm_pos0=DEFAULT_GM_POS0, wall=None,
                 scale=50):
        self.length = length
        self.width = width
        self.n_states = self.length * self.width
        self.goal = goal
        self.gm_pos0 = gm_pos0
        self.pos0 = DEFAULT_POS0 if pos0 is None else pos0
        self.gm_pos0 = gm_pos0
        self.wall = DEFAULT_WALL if wall is None else wall

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(1)

        self.pos = []
        for pos in self.pos0:
            self.pos += [[pos[0], pos[1]]]
        self.gm_pos = [self.gm_pos0[0], self.gm_pos0[1]]

        # render
        self.scale = scale
        pygame.init()

        self.steps = 0

        # meta
        self.win_history = [deque([0]), deque([0])]
        self.win_rate = [0., 0.]
        self.winner = False

        # balance state
        self.locked = False
        self.locked_state = 0

    def step(self, actions):

        # save old pos
        old_pos = []
        for pos in self.pos:
            old_pos += [(pos[0], pos[1])]

        # move
        for i, action in enumerate(actions):
            if i < 2:
                if action == 0:
                    self.pos[i][0] = (self.pos[i][0] + 1) % self.width
                elif action == 1:
                    self.pos[i][0] = (self.pos[i][0] - 1) % self.width
                elif action == 2:
                    self.pos[i][1] = (self.pos[i][1] + 1) % self.length
                elif action == 3:
                    self.pos[i][1] = (self.pos[i][1] - 1) % self.length

        # collision
        collisions = []
        for i, pos in enumerate(self.pos):
            collision = (pos[0], pos[1]) in self.wall or (pos[0] == self.gm_pos[0] and pos[1] == self.gm_pos[1])
            collisions += [collision]
            if collision:
                pos[0] = old_pos[i][0]
                pos[1] = old_pos[i][1]

        # terminal state
        goals = []
        for pos in self.pos:
            goals += [pos[0] == self.goal[0] and pos[1] == self.goal[1]]
        terminal = reduce((lambda x, y: x and y), goals)

        # Compute win rate
        if goals[0] and not self.winner:
            self.winner = True
            self.win_history[0].appendleft(1)
            # if both players win simultaneously
            if goals[1]:
                self.win_history[1].appendleft(1)
            else:
                self.win_history[1].appendleft(0)
        if goals[1] and not self.winner:
            self.winner = True
            self.win_history[0].appendleft(0)
            self.win_history[1].appendleft(1)
        if self.steps >= 25 and not self.winner:
            self.win_history[0].appendleft(0)
            self.win_history[1].appendleft(0)
        if len(self.win_history[0]) > 10:
            self.win_history[0].pop()
        if len(self.win_history[1]) > 10:
            self.win_history[1].pop()
        self.win_rate[0] = np.average(self.win_history[0])
        self.win_rate[1] = np.average(self.win_history[1])

        # get reward
        rewards = []
        for i, _ in enumerate(self.pos):
            if goals[i]:
                rewards += [1.]
            elif collisions[i]:
                rewards += [-.5]
            else:
                rewards += [0.]

        # get states
        states = []
        for pos in self.pos:
            states += [self.__state(pos)]

        # game master
        balance_reward = [0.]
        balance_state = self.__balance_state()

        if len(actions) > 2:

            # save old pos
            gm_old_pos = (self.gm_pos[0], self.gm_pos[1])

            # move
            gm_action = actions[2]
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
            gm_collision_player = (self.pos[0][0] == self.gm_pos[0] and self.pos[0][1] == self.gm_pos[1]) or (
                    self.pos[1][0] == self.gm_pos[0] and self.pos[1][1] == self.gm_pos[1])
            if gm_collision_wall or gm_collision_player:
                self.gm_pos = [gm_old_pos[0], gm_old_pos[1]]

            # get balance reward
            # player 0 better than player 1
            if balance_state == 1:
                if goals[0] and not goals[1]:
                    balance_reward = [-1.]
                elif goals[1]:
                    balance_reward = [1.]
                elif gm_collision_wall:
                    balance_reward = [-.5]
            # player 1 better than player 0
            elif balance_state == 2:
                if goals[0]:
                    balance_reward = [1.]
                elif not goals[0] and goals[1]:
                    balance_reward = [-1.]
                elif gm_collision_wall:
                    balance_reward = [-.5]
            # no one is better
            elif balance_state == 0:
                if self.gm_pos[0] == self.gm_pos0[0] and self.gm_pos[1] == self.gm_pos0[1]:
                    balance_reward = [1.]
                elif goals[0] and goals[1]:
                    balance_reward = [1.]
                elif gm_collision_wall:
                    balance_reward = [-.5]
                else:
                    balance_reward = [-1.]

        # get gm state
        gm_state = self.__gm_state(self.gm_pos, self.__balance_state())

        # get joint_state
        # joint_state = self.__joint_state([states], self.__state(self.gm_pos))

        # get gm joint_state
        gm_joint_state = self.__joint_state(states, gm_state)

        states += [gm_joint_state]

        # return step results
        return states, rewards + balance_reward, terminal, None

    def reset(self):

        self.winner = False

        self.steps = 0

        # set pos to pos0
        self.pos = []
        for pos in self.pos0:
            self.pos += [[pos[0], pos[1]]]

        # set gm_pos to gm_pos0
        self.gm_pos = [self.gm_pos0[0], self.gm_pos0[1]]

        # get states
        states = []
        for pos in self.pos:
            states += [self.__state(pos)]

        # get gm state
        gm_state = self.__gm_state(self.gm_pos, self.__balance_state())

        # get joint_state
        # joint_state = self.__joint_state([states], self.__state(self.gm_pos))

        # get gm joint_state
        gm_joint_state = self.__joint_state(states, gm_state)

        states += [gm_joint_state]

        # return reset state
        return states

    def render(self, mode='human'):

        # set the height and width of the screen
        size = [self.scale * self.length, self.scale * self.width]
        screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Maze Multi Player")

        # clear the screen and set the screen background
        balance_state = self.__balance_state()
        if balance_state == 1:
            screen.fill(YELLOW)
        elif balance_state == 2:
            screen.fill(PURPLE)
        else:
            screen.fill(WHITE)

        # Draw goal
        pygame.draw.rect(screen, GREEN, [self.goal[0] * self.scale, self.goal[1] * self.scale, self.scale, self.scale])

        # Draw walls
        for wall in self.wall:
            pygame.draw.rect(screen, BLACK, [wall[0] * 50, wall[1] * 50, 50, 50])

        # draw players
        for pos in self.pos:
            pygame.draw.rect(screen, BLUE, [pos[0] * self.scale, pos[1] * self.scale, self.scale, self.scale])

        # draw game master
        pygame.draw.rect(screen, RED,
                         [self.gm_pos[0] * self.scale, self.gm_pos[1] * self.scale, self.scale, self.scale])

        pygame.display.flip()

    def __state(self, v):
        return v[0] + self.length * v[1]

    def __gm_state(self, v, balance_state):
        return self.__state(v) + self.n_states * balance_state

    def __joint_state(self, s, gm_s):
        return s[0] + self.n_states * (s[1] + self.n_states * gm_s)

    def __balance_state(self):
        if self.locked:
            return self.locked_state
        return 0 if abs(self.win_rate[0] - self.win_rate[1]) <= .1 else 1 if self.win_rate[0] > self.win_rate[1] else 2

    def lock_balance_state(self, state):
        self.locked = True
        self.locked_state = state

    def free_balance_state(self):
        self.locked = False
