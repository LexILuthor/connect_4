import numpy as np


def is_winning(matrix, last_move_column, empty=0, red=-1, yellow=1):
    # note this function expect that last_move_column is a legal value and that row is not empty

    # find the row of the last move
    cell_value = empty
    last_move_row = -1
    while cell_value == empty:
        last_move_row = last_move_row + 1
        cell_value = matrix[last_move_row][last_move_column]
    # first we check if below the last move there are three gettoni of the same color (we do it only if we are above the third row)
    ################################################## need to ad a check that the last row has at least 3 other rows below
    if last_move_row < len(matrix) - 3:
        if matrix[last_move_row + 1][last_move_column] == cell_value:
            # if the sum of the following three values is tree then they are all of the same color
            tmp_sum = sum(matrix[last_move_row + 1:last_move_row + 4, last_move_column])
            if abs(tmp_sum) == 3:
                return True

    # second we check the positive diagonal
    counter = 0
    current_row = last_move_row
    current_column = last_move_column
    # checking the following items on the diagonal
    while next_cell_on_the_diagonal(matrix, current_row, current_column, direction=1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = next_cell_on_the_diagonal(matrix, current_row, current_column, direction=1)

    # checking the preceding items on the diagonal
    current_row = last_move_row
    current_column = last_move_column
    while prev_cell_on_the_diagonal(matrix, current_row, current_column, direction=1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = prev_cell_on_the_diagonal(matrix, current_row, current_column, direction=1)
    if counter >= 3:
        return True

    # third we check the negative diagonal
    counter = 0
    current_row = last_move_row
    current_column = last_move_column
    # checking the following items on the diagonal
    while next_cell_on_the_diagonal(matrix, current_row, current_column, direction=-1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = next_cell_on_the_diagonal(matrix, current_row, current_column,
                                                                       direction=-1)

    # checking the preceding items on the diagonal
    current_row = last_move_row
    current_column = last_move_column
    while prev_cell_on_the_diagonal(matrix, current_row, current_column, direction=-1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = prev_cell_on_the_diagonal(matrix, current_row, current_column,
                                                                       direction=-1)
    if counter >= 3:
        return True

    # fourth we check on the same row
    counter = 0
    current_column = last_move_column - 1

    # check on the left
    while current_column >= 0:
        if matrix[last_move_row][current_column] == cell_value:
            current_column = current_column - 1
            counter = counter + 1
        else:
            break

    # check on the right
    current_column = last_move_column + 1
    while current_column <= len(matrix[0]):
        if matrix[last_move_row][current_column] == cell_value:
            current_column = current_column + 1
            counter = counter + 1
        else:
            break

    if counter >= 3:
        return True

    return False


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
