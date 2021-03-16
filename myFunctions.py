import numpy as np
import matplotlib.pyplot as plt

import secondary_Functions as secFun


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


#----------------------------------------------------------------------------------------------------------
#   TO DO:
#
#   

#----------------------------------------------------------------------------------------------------------
def play_a_game(Q, epsilon=0.1, number_of_rows=6, number_of_columns=7,
                rewards_Wi_Lo_Dr_De=(1, -1, -0.5, 0), print_stuff=False):
    # "rewards_Wi_Lo_Dr_De" is the vector containing respectively the reward for a winning action, losing action,
    # draw action, nothing happens action

    # initialize an empty board
    board = np.zeros([number_of_rows, number_of_columns]).astype(int)

    empty = 0
    agent_color = 1
    ambient_color = -1

    # states reached during this game

    # we will be able to recognize the terminal states because r will be != 0
    SA_intermediate_state = []
    r = []
    S_prime = []

    while True:

        # agent makes a move
        agent_move_row, agent_move_column = secFun.agent_move_following_epsilon_Q(board, agent_color, epsilon, Q, empty)
        SA_intermediate_state.append(board)
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
    # graphic stuff
    if print_stuff:
        print_board(board, empty)
    # --------------------------------------------------------------------------------------------------------------

    return SA_intermediate_state, r, S_prime
