import numpy as np

import myFunctions as myFun
import neural_network as nn


def main():
    number_of_games = 100
    n_rows = 7
    n_columns = 6
    Q = nn.initialize_NN(n_rows, n_columns)

    # Memory of the agent ----------------------------------------------------------------------------------------------
    memory_size = 1000  # Let's set the memory capacity

    # ------------------------------------------------------------------------------------------------------------------

    Q = myFun.play_and_learn(number_of_games, memory_size, Q, n_rows, n_columns)


if __name__ == '__main__':
    main()
