import copy
import time
import numpy as np
import matplotlib.pyplot as plt


class GamePlay:

    def __init__(self, game, player, n_eps=100, n_steps=50, gen=False, render=False, plot=False, sleep=0, plot_id=0,
                 p_color='b', p_rate=1, p_name=None):
        """

        :param game:
        :param player:
        :param n_eps:
        :param n_steps:
        """
        self.n_eps = n_eps
        self.n_steps = n_steps
        self.game = game
        self.player = player
        self.gen = gen
        self.population = [] if self.gen else None
        self.render = render
        self.sleep = sleep
        # simulation
        self.ep = 0
        self.step = 0
        # plot
        self.q_history = []
        self.r_history = []
        self.q_ep = []
        self.r_ep = []
        self.ep_list = []
        self.plot = plot
        self.plot_id = plot_id
        self.p_color = p_color
        self.p_rate = p_rate
        self.p_name = p_name

    def train(self):
        """

        :return:
        """
        n_players = len(self.player)
        a = [None] * n_players
        s1 = self.game.reset()
        for i, p in enumerate(self.player):
            p.check_state(s1[i])
        ep = 0
        while ep < self.n_eps:
            ep += 1
            # print(ep)
            t = False
            step = 0
            if self.render:
                self.game.render()
                time.sleep(self.sleep)
            while not t and step < self.n_steps:
                step += 1
                for i, p in enumerate(self.player):
                    a[i] = p.get_action(s1[i])
                s0 = s1
                s1, r, t, _ = self.game.step(tuple(a))
                for i, p in enumerate(self.player):
                    p.update(s0[i], s1[i], a[i], r[i], t)
                if self.render:
                    self.game.render()
                    time.sleep(self.sleep)
                if self.plot:
                    self.q_ep += [np.max(self.player[self.plot_id].q[s1[self.plot_id]])]
                    # self.r_ep += [r[self.plot_id]]
            if self.gen:
                # for p in self.player:
                #    self.population.append((copy.deepcopy(p)))
                self.population.append((copy.deepcopy(self.player[-1])))
            if self.plot:
                self.q_history += [np.average(self.q_ep)]
                self.q_ep = []
                # self.r_history += [np.sum(self.r_ep)]
                # self.r_ep = []
                self.ep_list += [ep]
                if ep == self.p_rate:
                    plt.clf()
                if ep % self.p_rate == 0:
                    plt.plot(self.ep_list, self.q_history, color=self.p_color)
                    # plt.plot(self.ep_list, self.r_history, color=self.p_color, linestyle='dashed')
                    plt.xlabel('Episode')
                    plt.ylabel('Average Q')
                    plt.title('Learning Performance')
                    plt.draw()
                    plt.pause(0.01)
            s1 = self.game.reset()
        if self.p_name is not None:
            # plt.savefig(self.p_name + '.png')
            f = open(self.p_name + '_q_history.txt', 'w')
            f.write(str(self.q_history))
        return s1

    def get_population(self):
        return self.population
