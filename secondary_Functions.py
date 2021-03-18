import numpy as np
import copy
from random import *

import neural_network as nn


def remove_one_experience(S, a, r, S_prime):
    slot_to_be_removed = randint(0, len(r) - 1)
    S.pop(slot_to_be_removed)
    a.pop(slot_to_be_removed)
    r.pop(slot_to_be_removed)
    S_prime.pop(slot_to_be_removed)


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


#---------------------------------------------------------------------------------------------------
def states_that_can_be_reached_from(board, color):
    possible_states = []
    for i in range(len(board[0])):
        row = get_last_occupied_row_in_column(board, i, empty=0)
        if row > 0:
            possible_states.append(copy.copy(board))
            possible_states[-1][row - 1][i] = color
    return possible_states

#---------------------------------------------------------------------------------------------------
# This the implementation of a random environment
def ambient_move(board, ambient_color, empty=0):
    ambient_move_row, ambient_move_column = random_move(board, ambient_color, empty)
    return ambient_move_row, ambient_move_column

#---------------------------------------------------------------------------------------------------
# Function that makes the agent play according to the epsilon-greedy strategy
# Takes in input the current state (board), the agent color (=1), the epsilon parameter,
# the NN Q and the color of empty "pixels" (=0)
def agent_move_following_epsilon_Q(board, agent_color, epsilon, Q, empty=0):
    # decide if we play randomly (epsilon) or following the value function (1-epsilon)
    if random() < epsilon:
        print("DEBUGGING: agent plays randomly")
        # the agent play randomly
        agent_move_row, agent_move_column = copy.deepcopy(random_move(board, agent_color, empty))
        return agent_move_row, agent_move_column
    else:
        # the agent makes his move based on the value function
        available_actions = []
        for i in range(len(board[0])):
        # check whether the upper slot is empty
            if board[0][i] == empty:
                available_actions.append(int(i))
        # let's invoke the NN
        action_values = nn.Q_eval(Q, board)
        # we choose the max among the available ones
        # first we need to create a mask before applying argmax
        m = np.ones(action_values.size, dtype=bool)
        m[available_actions] = False
        masked_action_values = np.ma.array(action_values, mask=m)
        agent_move_column = np.argmax(masked_action_values)
        agent_move_row = get_last_occupied_row_in_column(board, agent_move_column, empty) - 1
        return agent_move_row, agent_move_column

#--------------------------------------------------------------------------------------------------------
def random_move(board, color_of_player, empty=0):
    # expect a non full board as input
    extraction_list = []
    # collect the indexes of the columns wich are not full
    for i in range(len(board[0])):
        # check whether the upper slot is empty
        if board[0][i] == empty:
            extraction_list.append(int(i))
    # random extract from the available columns
    extraction_index = randint(0, len(extraction_list) - 1)
    column_move = extraction_list[extraction_index]
    row_move = get_last_occupied_row_in_column(board, column_move, empty) - 1
    return row_move, column_move




#----------------------------------------------------------------------------------------------------------
def get_last_occupied_row_in_column(board, column, empty=0):
    # !!Warning: it returns len(board) if the column is empty !!
    # !!Warning len(board) is out of bound!!

    row = len(board)
    while row > 0 and board[row - 1][column] != empty:
        row = row - 1
    return row

#---------------------------------------------------------------------------------------------------------
def is_full(board, empty=0):
    # np.count_nonzero(board[0] == 0) it seems counterintuitive but this line actually counts how many zeros
    # there are in board[0]
    if np.count_nonzero(board[0] == empty) == 0:
        return True
    return False

#--------------------------------------------------------------------------------------------------------
def is_winning(board, last_move_row=-2, last_move_column=0, empty=0, red=-1, yellow=1):
    # note this function expect that last_move_column is a legal value and that row is not empty
    # if last_move_row=-2 it means we do not know the last_move_row and we have to find it using the function
    # "get_last_occupied_row_in_column(board, last_move_column, empty)"

    # find the row of the last move
    if last_move_row == -2:
        last_move_row = get_last_occupied_row_in_column(board, last_move_column, empty)

    cell_value = board[last_move_row][last_move_column]
    # first we check if below the last move there are three gettoni of the same color
    # (we do it only if we are above the third row)

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
    while next_cell_on_the_diagonal(board, current_row, current_column, direction=1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = next_cell_on_the_diagonal(board, current_row, current_column, direction=1)

    # checking the preceding items on the diagonal
    current_row = last_move_row
    current_column = last_move_column
    while prev_cell_on_the_diagonal(board, current_row, current_column, direction=1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = prev_cell_on_the_diagonal(board, current_row, current_column, direction=1)
    if counter >= 3:
        return True

    # third we check the negative diagonal
    counter = 0
    current_row = last_move_row
    current_column = last_move_column
    # checking the following items on the diagonal
    while next_cell_on_the_diagonal(board, current_row, current_column, direction=-1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = next_cell_on_the_diagonal(board, current_row, current_column, direction=-1)

    # checking the preceding items on the diagonal
    current_row = last_move_row
    current_column = last_move_column
    while prev_cell_on_the_diagonal(board, current_row, current_column, direction=-1)[0] == cell_value:
        counter = counter + 1
        waste, current_row, current_column = prev_cell_on_the_diagonal(board, current_row, current_column, direction=-1)
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
