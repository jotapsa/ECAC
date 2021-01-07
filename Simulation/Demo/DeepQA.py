from ast import literal_eval

from Game.BalanceableMaze import *
from Simulation.Api.GamePlay import *
from Player.MixedAgent import *
from Player.DeepQAgent import *


def main():

    N_EPS = 16000
    N_STEPS = 25

    game = BalanceableMaze()
    p = DeepQAgent(1, 4)
    game_play = GamePlay(game, [p], n_eps=N_EPS, n_steps=N_STEPS, plot=True, p_rate=800,
                         p_name='balanceable_maze_gm')
    game_play.train()


if __name__ == "__main__":
    main()
