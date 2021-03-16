import numpy as np
import matplotlib.pyplot as plt
import random

import secondary_Functions as secFun


# ----------------------------------------------------------------------------------------------------------------------
#   TO DO:
#
#   

# ----------------------------------------------------------------------------------------------------------------------

def play_and_learn(number_of_games, memory_size, Q):
    S = []
    a = []
    r = []
    S_prime = []

    for i in range(number_of_games):
        S_tmp, a_tmp, r_tmp, S_prime_tmp = play_a_game(Q, S, a, r, S_prime)

        while len(r) + len(r_tmp) >= memory_size:  # Check if the memory is already full
            # remove a (random) element from the tree lists N.
            secFun.remove_one_experience(S, a, r, S_prime)

        # we put the experience from this last game with the overall experience
        S.extend(S_tmp)
        a.extend(a_tmp)
        r.extend(r_tmp)
        S_prime.extend(S_prime_tmp)

    return Q


def play_a_game(Q, S, a, r, S_prime, epsilon=0.1, number_of_rows=6, number_of_columns=7,
                rewards_Wi_Lo_Dr_De=(1, -1, -0.5, 0), print_stuff=False):
    # "rewards_Wi_Lo_Dr_De" is the vector containing respectively the reward for a winning action, losing action,
    # draw action, nothing happens action

    # initialize an empty board
    board = np.zeros([number_of_rows, number_of_columns]).astype(int)

    empty = 0
    agent_color = 1
    ambient_color = -1

    while True:

        # agent makes a move
        S.append(board)

        agent_move_row, agent_move_column = secFun.agent_move_following_epsilon_Q(board, agent_color, epsilon, Q, empty)

        a.append(agent_move_column)

        # --------------------------------------------------------------------------------------------------------------
        # graphic stuff
        if print_stuff:
            print_board(board, empty)
        # --------------------------------------------------------------------------------------------------------------

        # check if the agent won
        if secFun.is_winning(board, agent_move_column, agent_move_row, empty):
            # Since we are in a terminal state S_prime is not important
            S_prime.append(board)
            r.append(rewards_Wi_Lo_Dr_De[0])
            break

        # check if board is full
        if secFun.is_full(board, empty):
            S_prime.append(board)
            r.append(rewards_Wi_Lo_Dr_De[2])
            break

        # ambient makes a (random) move
        ambient_move_row, ambient_move_column = secFun.ambient_move(board, ambient_color, empty)

        # check if ambient won
        if secFun.is_winning(board, ambient_move_column, ambient_move_row, empty):
            S_prime.append(board)
            r.append(rewards_Wi_Lo_Dr_De[1])
            break

        # check if board is full
        if secFun.is_full(board, empty):
            S_prime.append(board)
            r.append(rewards_Wi_Lo_Dr_De[2])
            break

        # it was a "nothing happens" action
        S_prime.append(board)
        r.append(rewards_Wi_Lo_Dr_De[3])

        # --------------------------------------------------------------------------------------------------------------
        # here the turn ends (both the agent and the ambient have done their move)

        # sample a batch of 4 from (SA_intermediate_state, r, S_prime)

        # !should i remove the selected states?!

        dimension_of_the_batch = 4
        my_batch = [random.randrange(len(r)) for i in range(dimension_of_the_batch)]

        selected_S = [S[i] for i in my_batch]
        selected_a = [a[i] for i in my_batch]
        selected_r = [r[i] for i in my_batch]
        selected_S_prime = [S_prime[i] for i in my_batch]

        # train_my_NN(Q):

        # --------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # graphic stuff
    if print_stuff:
        print_board(board, empty)
    # ------------------------------------------------------------------------------------------------------------------

    return S, a, r, S_prime


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
