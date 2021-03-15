import numpy as np
import matplotlib.pyplot as plt
from random import *

import secondary_Functions as secFun


def print_board(board, empty=0, red=-1, yellow=1):
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


def play_a_game(value_function, epsilon=0.1, number_of_rows=6, number_of_columns=7, print_stuff=False):
    # initialize an empty board
    board = np.zeros((number_of_rows, number_of_columns), dtype=np.int8)

    empty = 0
    agent_color = 1
    ambient_color = -1

    # states reached during this game
    states_reached = []

    agent_win = False
    is_a_draw = False

    while True:

        # agent makes a move
        agent_move_row, agent_move_column = secFun.agent_move_following_epsilon_value_function(board, agent_color,
                                                                                               epsilon, value_function,
                                                                                               empty)
        states_reached.append(board)
        # --------------------------------------------------------------------------------------------------------------
        # graphic stuff
        if print_stuff:
            print_board(board, empty)
        # --------------------------------------------------------------------------------------------------------------

        # check if the agent won
        if secFun.is_winning(board, agent_move_column, agent_move_row, empty):
            agent_win = True
            break

        # check if board is full
        if secFun.is_full(board, empty):
            is_a_draw = True
            break

        # ambient makes a (random) move
        ambient_move_row, ambient_move_column = secFun.ambient_move(board, ambient_color, empty)
        states_reached.append(board)

        # check if ambient won
        if secFun.is_winning(board, ambient_move_column, ambient_move_row, empty):
            agent_win = False
            break

        # check if board is full
        if secFun.is_full(board, empty):
            is_a_draw = True
            break

    if agent_win:
        return states_reached, 1
    elif is_a_draw:
        return states_reached, 0
    else:
        return states_reached, -1


def update_value_function(value_function, states_reached, result):
    reward = result
    for state in states_reached:
        value_of_state = value_function.get(np.ndenumerate(state), None)
        if value_of_state is None:
            value_function[np.ndenumerate(state)] = reward
        else:
            value_function[np.ndenumerate(state)] = value_function[np.ndenumerate(state)] + reward
