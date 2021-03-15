import numpy as np
import copy
from random import *





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
        row = get_last_occupied_row_in_column(board, i, empty=0)
        if row > 0:
            possible_states.append(copy.copy(board))
            possible_states[-1][row - 1][i] = color
    return possible_states





def agent_move_following_epsilon_value_function(board, agent_color, epsilon, value_function, empty=0,
                                                default_value_function=0):
    # if the state we are considering has no vale we set it to default_value_function

    # decide if we play randomly (epsilon) or following the value function (1-epsilon)
    if random() < epsilon:
        # the agent play randomly
        agent_move_row, agent_move_column = random_move(board, agent_color, empty)
        return agent_move_row, agent_move_column
    else:
        # the agent makes his move based on the value function
        possible_states = states_that_can_be_reached_from(board, agent_color)
        max_value = -np.inf
        possible_choices = []
        for state in possible_states:
            value_of_state = value_function.get(np.ndenumerate(state), None)
            if value_of_state is None:
                value_of_state = default_value_function

            if value_of_state > max_value:
                del possible_choices[:]
                possible_choices.append(state)
                max_value = value_of_state

            # this elif is actually useless because the confrontation between float can't be equal (this is how it works in C++)
            elif value_of_state == max_value:
                possible_choices.append(state)
        chosen_state = possible_choices[randint(0, len(possible_choices) - 1)]
        difference_matrix = board - chosen_state
        agent_move_row, agent_move_column = np.nonzero(difference_matrix)
        agent_move_row = agent_move_row[0]
        agent_move_column = agent_move_column[0]
        board[agent_move_row][agent_move_column] = agent_color
        return agent_move_row, agent_move_column


def random_move(board, color_of_player, empty=0):
    # expect a non full board as input
    extraction_list = []
    for i in range(len(board[0])):
        if board[0][i] == empty:
            extraction_list.append(int(i))
    extraction_index = randint(0, len(extraction_list) - 1)
    column_move = extraction_list[extraction_index]
    row_move = get_last_occupied_row_in_column(board, column_move, empty) - 1
    board[column_move][row_move] = color_of_player
    return row_move, column_move


def get_last_occupied_row_in_column(board, column, empty=0):
    # !!Warning: it returns len(board) if the column is empty !!
    # !!Warning len(board) is out of bound!!

    row = len(board)
    while row > 0 and board[row - 1][column] != empty:
        row = row - 1
    return row
