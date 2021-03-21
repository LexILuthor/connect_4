import numpy as np
import matplotlib.pyplot as plt
import random
import copy

import secondary_Functions as secFun
import neural_network as nn


# a function where the NN is first trained with epsilon_greedy and then evaluated with epsilon = 0
def evaluate_performance(Q, Q_ambient, name_of_the_model, number_of_evaluations, number_of_games, memory_size, n_rows,
                         n_columns, epsilon, play_as_second):
    probability_of_success = [0.5]
    total_games_played = [0]
    for i in range(number_of_evaluations):
        print("evaluation number " + str(i))

        number_of_games_during_evaluation = number_of_games
        wins, draw = play_and_learn(number_of_games_during_evaluation, memory_size, Q, Q_ambient, name_of_the_model,
                                    n_rows, n_columns, epsilon=epsilon, play_as_second=play_as_second)

        probability_of_success.append((wins + draw) / number_of_games_during_evaluation)
        total_games_played.append(total_games_played[-1] + number_of_games)

    secFun.plot_performances(total_games_played, probability_of_success)


def play_and_learn(number_of_games, memory_size, Q, Q_ambient, name_of_the_model, n_rows, n_columns, epsilon=0,
                   play_as_second=False):
    SA_intermediate_state = []
    r = []
    S_prime = []
    SA_intermediate_state_P2 = []
    r_P2 = []
    S_prime_P2 = []
    wins = 0
    draw = 0
    for i in range(number_of_games):

        if int(i) % 4 == 0:
            prints = False
        else:
            prints = False

        agent_won, game_draw = play_a_game(Q, Q_ambient, SA_intermediate_state, r, S_prime, SA_intermediate_state_P2,
                                           r_P2, S_prime_P2, n_rows, n_columns, epsilon,
                                           print_stuff=prints, play_as_second=play_as_second)
        if int(i) + 1 % 100 == 0:
            nn.save_NN(Q, name_of_the_model)
            nn.save_NN(Q_ambient, name_of_the_model + "-player_2")

        if agent_won:
            wins = wins + 1

        if game_draw:
            draw = draw + 1

        while len(r) >= memory_size:  # Check if the memory is already full
            # remove a (random) element from the tree lists N.
            secFun.remove_one_experience(SA_intermediate_state, r, S_prime, SA_intermediate_state_P2,
                                         r_P2, S_prime_P2, random=True)

    nn.save_NN(Q, name_of_the_model)
    nn.save_NN(Q_ambient, name_of_the_model + "-player_2")

    return wins, draw


def play_a_game(Q, Q_ambient, SA_intermediate_state, r, S_prime, SA_intermediate_state_P2, r_P2, S_prime_P2,
                number_of_rows=6, number_of_columns=7, epsilon=0.1, rewards_Wi_Lo_Dr_De=(10, -10, 3, 0),
                print_stuff=False, play_as_second=False):
    # "rewards_Wi_Lo_Dr_De" is the vector containing respectively the reward for a winning action, losing action,
    # draw action, nothing happens action

    empty = 0

    # initialize an empty board
    board = np.zeros([number_of_rows, number_of_columns]).astype(int)
    agent_color = 1
    ambient_color = -1

    # ------- !!!!!!!!!!!!!  If play as second is true the agent will play with vale -1 as second player !!!!!!!!!------

    if play_as_second:
        # let the ambient do the first  move at random
        secFun.random_move(board, ambient_color, empty)
        agent_color = -1
        ambient_color = 1

    # -------------------------------------------------------------------------------------------------------------

    agent_won = False
    game_draw = False

    number_of_moves = 0
    SA_intermediate_state_P2.append(np.matrix.copy(board))

    while True:

        # agent makes a move
        agent_move_row, agent_move_column = secFun.agent_move_following_epsilon_Q(board, agent_color, epsilon, Q, empty)
        SA_intermediate_state.append(np.matrix.copy(board))

        # --------------------------------------------------------------------------------------------------------------
        # graphic stuff
        if print_stuff:
            print_board(board, empty)
        # --------------------------------------------------------------------------------------------------------------

        # check if the agent won
        if secFun.is_winning(board, agent_move_column, agent_move_row, empty):
            # Since we are in a terminal state S_prime is not important
            S_prime.append(np.matrix.copy(board))
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[0]))
            S_prime_P2.append(np.matrix.copy(board))
            r_P2.append((copy.copy(rewards_Wi_Lo_Dr_De[1])))
            # print_board(board, empty)  # -------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! <------- delete me
            agent_won = True
            break

        # check if board is full
        if secFun.is_full(board, empty):
            S_prime.append(board)
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[2]))
            S_prime_P2.append(np.matrix.copy(board))
            r_P2.append((copy.copy(rewards_Wi_Lo_Dr_De[2])))
            # print_board(board, empty)  # -------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! <------- delete me
            game_draw = True
            break

        S_prime_P2.append(np.matrix.copy(board))
        r_P2.append((copy.copy(rewards_Wi_Lo_Dr_De[3])))
        # ambient makes a (random) move
        if len(S_prime_P2) != len(SA_intermediate_state_P2):
            print("here")
        ambient_move_row, ambient_move_column = secFun.ambient_move(board, Q_ambient, ambient_color, empty,
                                                                    epsilon=0.02)
        SA_intermediate_state_P2.append(np.matrix.copy(board))

        # check if ambient won
        if secFun.is_winning(board, ambient_move_column, ambient_move_row, empty):
            S_prime.append(np.matrix.copy(board))
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[1]))
            S_prime_P2.append(np.matrix.copy(board))
            r_P2.append((copy.copy(rewards_Wi_Lo_Dr_De[0])))
            # print_board(board, empty)  # -------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! <------- delete me
            break

        # check if board is full
        if secFun.is_full(board, empty):
            S_prime.append(board)
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[2]))
            S_prime_P2.append(np.matrix.copy(board))
            r_P2.append((copy.copy(rewards_Wi_Lo_Dr_De[2])))
            # print_board(board, empty)  # -------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! <------- delete me
            game_draw = True
            break

        # it was a "nothing happens" action
        S_prime.append(np.matrix.copy(board))
        r.append(copy.copy(rewards_Wi_Lo_Dr_De[3]))
        # here the turn ends (both the agent and the ambient have done their move)
        # ------------------------------------------------------------------------------------------------------------------
        # sample a batch of 4 from (SA_intermediate_state, r, S_prime)
        if number_of_moves % 8 == 0:
            batch_size = 8
            secFun.select_the_batch_and_train_the_NN(batch_size, Q, SA_intermediate_state, r, S_prime, agent_color)
            secFun.select_the_batch_and_train_the_NN(batch_size, Q_ambient, SA_intermediate_state_P2, r_P2, S_prime_P2,
                                                     ambient_color)
        number_of_moves += 1
        # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # graphic stuff
    if print_stuff:
        print_board(board, empty)
    # ------------------------------------------------------------------------------------------------------------------

    return agent_won, game_draw


def print_board(board, empty=0, red=-1):
    # yellow = 1
    empty_x = []
    empty_y = []
    red_x = []
    red_y = []
    yellow_x = []
    yellow_y = []
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == empty:
                empty_x.append(int(c))
                empty_y.append(-int(r))
            elif board[r][c] == red:
                red_x.append(int(c))
                red_y.append(-int(r))
            else:
                yellow_x.append(int(c))
                yellow_y.append(-int(r))

    point_dimension = 20
    shape_of_points = "H"
    fig, ax = plt.subplots()
    ax.set_xlim([-1, len(board[0])])
    ax.set_ylim(([-len(board), 1]))

    ax.scatter(empty_x, empty_y, color='grey', marker=shape_of_points, alpha=0.5, linewidths=point_dimension)
    ax.scatter(red_x, red_y, color='red', marker=shape_of_points, linewidths=point_dimension)
    ax.scatter(yellow_x, yellow_y, color='yellow', marker=shape_of_points, linewidths=point_dimension)
    fig.show()
