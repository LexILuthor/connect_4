import numpy as np

import myFunctions as myFun
import neural_network as nn


def main():
    number_of_games = 100
    Q = {}

    my_neural_network_Q = nn.initialize_the_NN()

    # Memory of the agent ----------------------------------------------------------------------------------------------
    memory_size = 1000  # Let's set the memory capacity

    # ------------------------------------------------------------------------------------------------------------------

    # a function that "plays number_of_games" times and train the neural network while playing
    # we will substitute Q with our neural network

    my_neural_network_Q = myFun.play_and_learn(number_of_games, memory_size, Q)


if __name__ == '__main__':
    main()
