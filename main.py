import numpy as np
import myFunctions as myFun
import neural_network as nn


def main():
    number_of_games = 60
    n_rows = 7
    n_columns = 6
    epsilon = 0.1

    #Q = nn.initialize_NN(n_rows, n_columns)
    Q = nn.load_NN(n_rows, n_columns)

    # Memory of the agent ----------------------------------------------------------------------------------------------
    memory_size = 1000  # Let's set the memory capacity

    # ------------------------------------------------------------------------------------------------------------------

    # wins = myFun.play_and_learn(number_of_games, memory_size, Q, n_rows, n_columns, epsilon)

    number_of_evaluations = 350
    myFun.evaluate_performance(Q, number_of_evaluations, number_of_games, memory_size, n_rows, n_columns, epsilon)

    nn.save_NN(Q)


if __name__ == '__main__':
    main()
