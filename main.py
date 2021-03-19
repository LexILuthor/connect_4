import numpy as np
import myFunctions as myFun
import neural_network as nn


def main():
    number_of_games = 10
    n_rows = 7
    n_columns = 6
    epsilon = 0.05
    name_of_the_model: str = "franz_NN_2"
    #Q = nn.initialize_NN(n_rows, n_columns)
    Q = nn.load_NN(name_of_the_model, n_rows, n_columns)

    # Memory of the agent ----------------------------------------------------------------------------------------------
    memory_size = 500  # Let's set the memory capacity

    # ------------------------------------------------------------------------------------------------------------------

    # wins, draw = myFun.play_and_learn(number_of_games, memory_size, Q, name_of_the_model, n_rows, n_columns, epsilon)

    number_of_evaluations = 10
    myFun.evaluate_performance(Q, name_of_the_model, number_of_evaluations, number_of_games, memory_size, n_rows,
                               n_columns, epsilon)

    nn.save_NN(Q, name_of_the_model)


if __name__ == '__main__':
    main()
