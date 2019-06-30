from Game.BalanceableMaze import *
from Simulation.Api.GamePlay import *
from Player.MixedAgent import *
from ast import literal_eval


def main():
    N_EPS = 10
    N_STEPS = 25

    print('Population Load')

    game = BalanceableMaze()
    population = []
    with open('balanceable_maze_population.txt', 'r') as f:
        for line in f:
            population += [MixedAgent(literal_eval(line))]

    print('Population Eval')

    population_eval = []

    for i, p in enumerate(population):
        game_play = GamePlay(game, [p], n_eps=N_EPS, n_steps=N_STEPS)
        game_play.train()
        population_eval += [game.win_rate]

    f = open('balanceable_maze_population_eval.txt', 'w')
    f.write(str(population_eval))


if __name__ == "__main__":
    main()
