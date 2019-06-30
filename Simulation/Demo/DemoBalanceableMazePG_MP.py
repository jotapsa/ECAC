from Game.BalanceableMazeMP import *
from Simulation.Api.GamePlay import *
from Player.ABWPL import *


def main():
    L_RATE = .1
    PI_L_RATE = L_RATE / 100
    Y = .9
    E_RATE = .1
    MIN_E_RATE = .0001
    N_EPS = 16000
    N_STEPS = 25

    print('Population Generation')

    game = BalanceableMazeMP()
    p = ABWPL(L_RATE, Y, E_RATE, game.action_space.n, PI_L_RATE, MIN_E_RATE)
    game_play = GamePlay(game, [p, p], n_eps=N_EPS, n_steps=N_STEPS, gen=True, plot=True, p_rate=800,
                         p_name='balanceable_maze_mp')
    game_play.train()
    population = game_play.get_population()
    f = open('balanceable_maze_mp_population.txt', 'w')
    for p in population:
        pi = {}
        for s in p.pi:
            pi[s] = p.pi[s].tolist()
        f.write(str(pi) + '\n')


if __name__ == "__main__":
    main()
