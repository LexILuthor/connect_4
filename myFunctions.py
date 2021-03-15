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


def play_a_game(value_function, epsilon=0.1, number_of_rows=6, number_of_columns=7):
    # initialize an empty board
    board = np.zeros((number_of_rows, number_of_columns), dtype=np.int8)

    empty = 0
    agent_color = 1
    ambient_color = -1

    agent_move_row, agent_move_column = secFun.agent_move_following_epsilon_value_function(board, agent_color, epsilon,
                                                                                           value_function, empty)
    if is_winning(board, agent_move_column, agent_move_row, empty):
        print("Agent win")
        return
    if is_full(board, empty):
        print("is full, it is a draw")
        return

    print_board(board, empty)


def is_full(board, empty=0):
    # np.count_nonzero(board[0] == 0) it seems counterintuitive but this line actually counts how many zeros there are in board[0]
    if np.count_nonzero(board[0] == empty) == 0:
        return True
    return False


def is_winning(board, last_move_column, last_move_row=-2, empty=0, red=-1, yellow=1):
    # note this function expect that last_move_column is a legal value and that row is not empty
    # if last_move_row=-2 it means we do not know the last_move_row and we have to find it using the function
    # "get_last_occupied_row_in_column(board, last_move_column, empty)"

    # find the row of the last move
    if last_move_row == -2:
        last_move_row = secFun.get_last_occupied_row_in_column(board, last_move_column, empty)

    cell_value = board[last_move_row][last_move_column]
    # first we check if below the last move there are three gettoni of the same color (we do it only if we are above the third row)

    if last_move_row < len(board) - 3:
        if board[last_move_row + 1][last_move_column] == cell_value:
            # if the sum of the following three values is tree then they are all of the same color
            tmp_sum = sum(board[last_move_row + 1:last_move_row + 4, last_move_column])
            if abs(tmp_sum) == 3:
                return True

    # second we check the positive diagonal
    counter = 0
    current_row = last_move_row
    current_column = last_move_column
    # checking the following items on the diagonal
    while secFun.next_cell_on_the_diagonal(board, current_row, current_column, direction=1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = secFun.next_cell_on_the_diagonal(board, current_row, current_column,
                                                                              direction=1)

    # checking the preceding items on the diagonal
    current_row = last_move_row
    current_column = last_move_column
    while secFun.prev_cell_on_the_diagonal(board, current_row, current_column, direction=1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = secFun.prev_cell_on_the_diagonal(board, current_row, current_column,
                                                                              direction=1)
    if counter >= 3:
        return True

    # third we check the negative diagonal
    counter = 0
    current_row = last_move_row
    current_column = last_move_column
    # checking the following items on the diagonal
    while secFun.next_cell_on_the_diagonal(board, current_row, current_column, direction=-1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = secFun.next_cell_on_the_diagonal(board, current_row, current_column,
                                                                              direction=-1)

    # checking the preceding items on the diagonal
    current_row = last_move_row
    current_column = last_move_column
    while secFun.prev_cell_on_the_diagonal(board, current_row, current_column, direction=-1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = secFun.prev_cell_on_the_diagonal(board, current_row, current_column,
                                                                              direction=-1)
    if counter >= 3:
        return True

    # fourth we check on the same row
    counter = 0
    current_column = last_move_column - 1

    # check on the left
    while current_column >= 0:
        if board[last_move_row][current_column] == cell_value:
            current_column = current_column - 1
            counter = counter + 1
        else:
            break

    # check on the right
    current_column = last_move_column + 1
    while current_column < len(board[0]):
        if board[last_move_row][current_column] == cell_value:
            current_column = current_column + 1
            counter = counter + 1
        else:
            break

    if counter >= 3:
        return True

    return False
