from Game.BalanceableMazeMP import *
from Player.MixedAgent import *
from Simulation.Api.GamePlay import *
from ast import literal_eval


def main():
    print('Population Load')

    population = []
    with open('balanceable_maze_mp_population.txt', 'r') as f:
        for line in f:
            population += [MixedAgent(literal_eval(line))]

    print('Game Master Load')

    pi_gm = {}
    with open('balanceable_maze_mp_game_master.txt', 'r') as f:
        for line in f:
            pi_gm = literal_eval(line)

    N_EPS = 100
    N_STEPS = 25

    print('Game Master Demo')

    game = BalanceableMazeMP()
    p1 = population[-1]
    p2 = population[800]
    gm = MixedAgent(pi_gm)
    game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS)
    game_play.train()

    for x in range(2):
        lst = [(i, p) for i, p in enumerate(game.win_rate_history[x])]
        for e in lst:
            print(str(e) + ' ', end='')
        print()


if __name__ == "__main__":
    main()
