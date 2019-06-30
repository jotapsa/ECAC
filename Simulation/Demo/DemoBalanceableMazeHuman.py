from Game.BalanceableMaze import *
from Player.MixedAgent import *
from Simulation.Api.GamePlay import *
from Player.HumanAgent import *
from ast import literal_eval


def main():

    print('Game Master Load')

    pi = {}
    with open('balanceable_maze_game_master.txt', 'r') as f:
        for line in f:
            pi = literal_eval(line)

    N_EPS = 50
    N_STEPS = 25

    game = BalanceableMaze()
    p = HumanAgent()
    gm = MixedAgent(pi)
    game_play = GamePlay(game, [p, gm], n_eps=N_EPS, n_steps=N_STEPS, render=True)
    game_play.train()


if __name__ == "__main__":
    main()
