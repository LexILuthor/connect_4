import numpy as np
import matplotlib.pyplot as plt
import random
import copy

import secondary_Functions as secFun
import neural_network as nn


# ----------------------------------------------------------------------------------------------------------------------
#   TO DO:
#
#   

# ----------------------------------------------------------------------------------------------------------------------

def play_and_learn(number_of_games, memory_size, Q, name_of_the_model, n_rows, n_columns, epsilon):
    SA_intermediate_state = []
    r = []
    S_prime = []
    wins = 0
    draw = 0
    for i in range(number_of_games):
        SA_intermediate_state_tmp, r_tmp, S_prime_tmp, agent_won, game_draw = play_a_game(Q, SA_intermediate_state, r,
                                                                                          S_prime,
                                                                                          n_rows, n_columns, epsilon,
                                                                                          print_stuff=False)
        if int(i) + 1 % 100 == 0:
            nn.save_NN(Q, name_of_the_model)

        if agent_won:
            wins = wins + 1

        if game_draw:
            draw = draw + 1

        while len(r) + len(r_tmp) >= memory_size:  # Check if the memory is already full
            # remove a (random) element from the tree lists N.
            secFun.remove_one_experience(SA_intermediate_state, r, S_prime, random=True)

        # we put the experience from this last game with the overall experience
        SA_intermediate_state.extend(SA_intermediate_state_tmp)
        r.extend(r_tmp)
        S_prime.extend(S_prime_tmp)

    nn.save_NN(Q, name_of_the_model)

    return wins, draw


def play_a_game(Q, SA_intermediate_state, r, S_prime, number_of_rows=6, number_of_columns=7, epsilon=0.1,
                rewards_Wi_Lo_Dr_De=(100, -100, -40, 0), print_stuff=False):
    # "rewards_Wi_Lo_Dr_De" is the vector containing respectively the reward for a winning action, losing action,
    # draw action, nothing happens action

    # initialize an empty board
    board = np.zeros([number_of_rows, number_of_columns]).astype(int)

    empty = 0

    # -------------------------------------------------------------------------------
    # WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! changed agent and ambient color!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!
    # WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! changed agent and ambient color!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!
    agent_color = 1
    ambient_color = -1
    # WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! changed agent and ambient color!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!
    # WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! changed agent and ambient color!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!
    # -------------------------------------------------------------------------------

    agent_won = False
    game_draw = False

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
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[1]))
            agent_won = True
            break

        # check if board is full
        if secFun.is_full(board, empty):
            S_prime.append(board)
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[2]))
            game_draw = True
            break

        # ambient makes a (random) move
        ambient_move_row, ambient_move_column = secFun.ambient_move(board, ambient_color, empty)

        # check if ambient won
        if secFun.is_winning(board, ambient_move_column, ambient_move_row, empty):
            S_prime.append(np.matrix.copy(board))
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[1]))
            break

        # check if board is full
        if secFun.is_full(board, empty):
            S_prime.append(board)
            r.append(copy.copy(rewards_Wi_Lo_Dr_De[2]))
            game_draw = True
            break

        # it was a "nothing happens" action
        S_prime.append(np.matrix.copy(board))
        r.append(copy.copy(rewards_Wi_Lo_Dr_De[3]))
        # here the turn ends (both the agent and the ambient have done their move)
        # ------------------------------------------------------------------------------------------------------------------
        # sample a batch of 4 from (SA_intermediate_state, r, S_prime)
        batch_size = 4
        secFun.select_the_batch_and_train_the_NN(batch_size, Q, SA_intermediate_state, r, S_prime, agent_color)
        # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # graphic stuff
    if print_stuff:
        print_board(board, empty)
    # ------------------------------------------------------------------------------------------------------------------

    return SA_intermediate_state, r, S_prime, agent_won, game_draw


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


# a function where the NN is first trained with epsilon_greedy and then evaluated with epsilon = 0
def evaluate_performance(Q, name_of_the_model, number_of_evaluations, number_of_games, memory_size, n_rows, n_columns,
                         epsilon):
    probability_of_success = [0.5]
    total_games_played = [0]
    for i in range(number_of_evaluations):
        print("evaluation number " + str(i))
        # wins, draw = play_and_learn(number_of_games, memory_size, Q, n_rows, n_columns, epsilon)

        number_of_games_during_evaluation = number_of_games
        wins, draw = play_and_learn(number_of_games_during_evaluation, memory_size, Q, name_of_the_model, n_rows,
                                    n_columns,
                                    epsilon=epsilon)

        probability_of_success.append((wins) / number_of_games_during_evaluation)
        total_games_played.append(total_games_played[-1] + number_of_games)

    secFun.plot_performances(total_games_played, probability_of_success)
