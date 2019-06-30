from Game.BalanceableMazeMP import *
from Player.MixedAgent import *
from Simulation.Api.GamePlay import *
from Player.ABWPL import *
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

    L_RATE = .1
    PI_L_RATE = L_RATE / 100
    Y = .9
    E_RATE = .1
    MIN_E_RATE = .0001
    N_STEPS = 25

    gm = ABWPL(L_RATE, Y, E_RATE, game.action_space.n, PI_L_RATE, MIN_E_RATE)

    for i1 in players:

        p1 = MixedAgent(population[i1].pi)

        for i2 in players:
            p2 = MixedAgent(population[i2].pi)

            print('vs Player ' + str(i1) + ' and ' + str(i2))

            print('state 0')
            N_EPS = 4000
            game.lock_balance_state(0)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=2, p_color='r',
                                 p_rate=1000)
            game_play.train()

            print('state 1')
            N_EPS = 16000
            game.lock_balance_state(1)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=2, p_color='g',
                                 p_rate=2000)
            game_play.train()

            print('state 2')
            game.lock_balance_state(2)
            game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=2, p_color='b',
                                 p_rate=2000)
            game_play.train()

    print('Game Master Demo')

    N_EPS = 5
    SLEEP = .2

    gm = MixedAgent(gm.pi)

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

    print('Game Master Store')

    f = open('balanceable_maze_mp_game_master.txt', 'w')
    pi = {}
    for s in gm.pi:
        pi[s] = gm.pi[s].tolist()
    f.write(str(pi) + '\n')


if __name__ == "__main__":
    main()
