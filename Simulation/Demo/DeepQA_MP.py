from Game.BalanceableMazeMP import *
from Simulation.Api.GamePlay import *
from Player.MixedAgent import *
from Player.DeepQAgent import *
from ast import literal_eval

def main():

    N_EPS = 16000
    N_STEPS = 25

    game = BalanceableMazeMP()
    p1 = DeepQAgent(1, 5)
    p2 = DeepQAgent(1, 5)
    gm = DeepQAgent(1, 5)
    game_play = GamePlay(game, [p1, p2, gm], n_eps=N_EPS, n_steps=N_STEPS, plot=True, p_rate=800,
                         p_name='balanceable_maze_gm')
    game_play.train()

if __name__ == "__main__":
    main()
