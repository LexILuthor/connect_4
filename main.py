import numpy as np
import myFunctions as myFun
import neural_network as nn


def main():
    number_of_games = 30
    number_of_evaluations = 40
    n_rows = 7
    n_columns = 6
    epsilon = 0.02
    memory_size = 500
    name_of_the_model: str = "franz_NN_2"

    train_player_2 = False

    if train_player_2:
        # Q = nn.initialize_NN(n_rows, n_columns)
        Q = nn.load_NN(name_of_the_model + "player_2", n_rows, n_columns)

        # Q_ambient = nn.initialize_NN(n_rows, n_columns)
        Q_ambient = nn.load_NN(name_of_the_model, n_rows, n_columns)
        name_of_the_model = name_of_the_model + "player_2"

    else:
        # Q = nn.initialize_NN(n_rows, n_columns)
        Q = nn.load_NN(name_of_the_model, n_rows, n_columns)

        #Q_ambient = nn.initialize_NN(n_rows, n_columns)
        Q_ambient = nn.load_NN(name_of_the_model + "player_2", n_rows, n_columns)

    # ------------------------------------------------------------------------------------------------------------------

    # wins, draw = myFun.play_and_learn(number_of_games, memory_size, Q,QA, name_of_the_model, n_rows, n_columns, epsilon)

    myFun.evaluate_performance(Q, Q_ambient, name_of_the_model, number_of_evaluations, number_of_games, memory_size,
                               n_rows, n_columns, epsilon, play_as_second=train_player_2)

    nn.save_NN(Q, name_of_the_model)


if __name__ == '__main__':
    main()
