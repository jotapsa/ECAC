from Game.BalanceableMazeMP import *
from Simulation.Api.GamePlay import *
from Player.MixedAgent import *
from Player.DeepQAgent import *
from ast import literal_eval

def main():
    print('Population Eval Load')

    population_eval = []
    with open('balanceable_maze_mp_population_eval.txt', 'r') as f:
        for line in f:
            population_eval = literal_eval(line)

    print('Player Selection')

    players = []
    for i in range(3, 11):
        p = round(i * 0.1, 1)
        players += [len(population_eval) - population_eval[::-1].index(p) - 1]

    players = [players[4], players[-3], players[-1]]

    print('Population Load')

    game = BalanceableMazeMP()
    population = []
    with open('balanceable_maze_mp_population.txt', 'r') as f:
        for line in f:
            population += [MixedAgent(literal_eval(line))]

    print('Game Master Training')

    N_STEPS = 25

    gm = DeepQAgent(1, 5)

    for i1 in players:
        p1 = MixedAgent(population[i1].pi)
        for i2 in players:
            p2 = MixedAgent(population[i2].pi)

            print('vs Player ' + str(i1) + ' and ' + str(i2))

            print('state 0')
            N_EPS = 4000
            game.lock_balance_state(0)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=2, p_color='r',
                                 p_rate=1000, p_name='balanceable_maze_mp_gm')
            game_play.train()

            print('state 1')
            N_EPS = 16000
            game.lock_balance_state(1)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=2, p_color='g',
                                 p_rate=2000, p_name='balanceable_maze_mp_gm')
            game_play.train()

            print('state 2')
            game.lock_balance_state(2)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=2, p_color='b',
                                 p_rate=2000, p_name='balanceable_maze_mp_gm')
            game_play.train()

    print('Game Master Demo')

    SLEEP = .2

    for i1 in players:
        p1 = MixedAgent(population[i1].pi)
        for i2 in players:
            p2 = MixedAgent(population[i2].pi)

            print('vs Player ' + str(i1) + ' and ' + str(i2))

            print('state 0')
            game.lock_balance_state(0)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, render=True, sleep=SLEEP)
            game_play.train()

            print('state 1')
            game.lock_balance_state(1)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, render=True, sleep=SLEEP)
            game_play.train()

            print('state 2')
            game.lock_balance_state(2)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, render=True, sleep=SLEEP)
            game_play.train()

if __name__ == "__main__":
    main()
