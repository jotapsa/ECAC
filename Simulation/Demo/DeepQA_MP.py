from Game.BalanceableMazeMP import *
from Simulation.Api.GamePlay import *
from Player.DeepQAgent import *


def main():
    print('Game Master Load')

    N_EPS = 16000
    N_STEPS = 25

    game = BalanceableMazeMP()
    p = DeepQAgent(1, 5)
    game_play = GamePlay(game, [p, p], n_eps=N_EPS, n_steps=N_STEPS, plot=True, p_rate=800,
                         p_name='balanceable_maze_gm')
    game_play.train()


if __name__ == "__main__":
    main()
