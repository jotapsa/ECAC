from Game.BalanceableMaze import *
from Player.MixedAgent import *
from Simulation.Api.GamePlay import *
from ast import literal_eval


def main():
    print('Population Load')

    population = []
    with open('balanceable_maze_population.txt', 'r') as f:
        for line in f:
            population += [MixedAgent(literal_eval(line))]

    print('Game Master Load')

    pi_gm = {}
    with open('balanceable_maze_game_master.txt', 'r') as f:
        for line in f:
            pi_gm = literal_eval(line)

    N_EPS = 100
    N_STEPS = 25

    print('Game Master Demo')

    game = BalanceableMaze()
    p = population[-1]
    gm = MixedAgent(pi_gm)
    game_play = GamePlay(game, [p, gm], n_eps=N_EPS, n_steps=N_STEPS)
    game_play.train()

    lst = [(i, p) for i, p in enumerate(game.win_rate_history)]
    for e in lst:
        print(str(e) + ' ', end='')


if __name__ == "__main__":
    main()
