import numpy as np
import copy


def next_cell_on_the_diagonal(matrix, current_row, current_column, direction):
    if direction == 1:
        # check we are in the boundaries
        if current_row == 0 or current_column == len(matrix[0]) - 1:
            # we return a value that in not in the matrix because we are outside the boundary
            return 1.5, None, None
        else:
            return matrix[current_row - 1][current_column + 1], current_row - 1, current_column + 1
    elif direction == -1:
        if current_row == 0 or current_column == 0:
            # we return a value that in not in the matrix because we are outside the boundary
            return 1.5, None, None
        else:
            return matrix[current_row - 1][current_column - 1], current_row - 1, current_column - 1


def prev_cell_on_the_diagonal(matrix, current_row, current_column, direction):
    if direction == 1:
        # check we are in the boundaries
        if current_row == len(matrix) - 1 or current_column == 0:
            # we return a value that in not in the matrix because we are outside the boundary
            return 1.5, None, None
        else:
            return matrix[current_row + 1][current_column - 1], current_row + 1, current_column - 1
    elif direction == -1:
        if current_row == len(matrix) - 1 or current_column == len(matrix[0]) - 1:
            # we return a value that in not in the matrix because we are outside the boundary
            return 1.5, None, None
        else:
            return matrix[current_row + 1][current_column + 1], current_row + 1, current_column + 1


def states_that_can_be_reached_from(board, color):
    possible_states = []
    for i in range(len(board[0])):
        row = get_last_row_in_column(board, i, empty=0)
        if row > 0:
            possible_states.append(copy.copy(board))
            possible_states[-1][row - 1][i] = color
    return possible_states


def get_last_row_in_column(board, column, empty=0):
    # !!Warning: it returns len(board) if the column is empty !!
    # !!Warning len(board) is out of bound!!

    row = len(board)
    while row > 0 and board[row - 1][column] != empty:
        row = row - 1
    return row
