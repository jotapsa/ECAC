from Game.BalanceableMaze import *
from Player.MixedAgent import *
from Simulation.Api.GamePlay import *
from Player.ABWPL import *
from ast import literal_eval


def main():
    print('Population Eval Load')

    population_eval = []
    with open('balanceable_maze_population_eval.txt', 'r') as f:
        for line in f:
            population_eval = literal_eval(line)

    print('Player Selection')

    players = []
    for i in range(1, 11):
        p = round(i * 0.1, 1)
        players += [len(population_eval) - population_eval[::-1].index(p) - 1]

    print('Population Load')

    game = BalanceableMaze()
    population = []
    with open('balanceable_maze_population.txt', 'r') as f:
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

    for player in players:
        print('vs Player ' + str(player))
        p = MixedAgent(population[player].pi)

        print('state 0')
        N_EPS = 2000
        game.lock_balance_state(0)
        game_play = GamePlay(game, [p, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=1, p_color='g', p_rate=500,
                             p_name='balanceable_maze_gm')
        game_play.train()

        print('state 1')
        N_EPS = 8000
        game.lock_balance_state(1)
        game_play = GamePlay(game, [p, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, plot_id=1, p_color='b', p_rate=500,
                             p_name='balanceable_maze_gm')
        game_play.train()

    print('Game Master Demo')

    N_EPS = 5
    SLEEP = .2

    gm = MixedAgent(gm.pi)

    for player in players:
        print('vs Player ' + str(player))
        p = MixedAgent(population[player].pi)

        print('state 0')
        game.lock_balance_state(0)
        game_play = GamePlay(game, [p, gm], n_eps=N_EPS, n_steps=N_STEPS, render=True, sleep=SLEEP)
        game_play.train()

        print('state 1')
        game.lock_balance_state(1)
        game_play = GamePlay(game, [p, gm], n_eps=N_EPS, n_steps=N_STEPS, render=True, sleep=SLEEP)
        game_play.train()

    print('Game Master Store')

    f = open('balanceable_maze_game_master.txt', 'w')
    pi = {}
    for s in gm.pi:
        pi[s] = gm.pi[s].tolist()
    f.write(str(pi) + '\n')


if __name__ == "__main__":
    main()
