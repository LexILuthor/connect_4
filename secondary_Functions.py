import numpy as np
import copy
from random import *
import matplotlib.pyplot as plt

import neural_network as nn
import myFunctions as myFun


def remove_one_experience(SA_intermediate_state, r, S_prime, random=False):
    if random:
        slot_to_be_removed = randint(0, len(r) - 1)
        if r[slot_to_be_removed] != 0:
            slot_to_be_removed = randint(0, len(r) - 1)
    else:
        slot_to_be_removed = 0

    SA_intermediate_state.pop(slot_to_be_removed)
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


def states_that_can_be_reached_from(board, color):
    possible_states = []
    for i in range(len(board[0])):
        row = get_last_occupied_row_in_column(board, i, empty=0)
        if row > 0:
            possible_states.append(copy.copy(board))
            possible_states[-1][row - 1][i] = color
    return possible_states


def ambient_move(board, Q_ambient, ambient_color, empty=0, epsilon=0):
    #  p(random move)=epsilon , p(NN)=1-epsilon
    if random() < epsilon:
        ambient_move_row, ambient_move_column = random_move(board, ambient_color, empty)
    else:
        ambient_move_row, ambient_move_column = agent_move_following_epsilon_Q(board, ambient_color, epsilon, Q_ambient,
                                                                               empty)

    # -----------------------------------------------------------------------------------------------------------------

    # manual move
    # ambient_move_column = int(input())

    # ambient_move_row = get_last_occupied_row_in_column(board, ambient_move_column, empty=0) - 1

    # board[ambient_move_row][ambient_move_column] = ambient_color

    # -----------------------------------------------------------------------------------------------------------------

    return ambient_move_row, ambient_move_column


def agent_move_following_epsilon_Q(board, agent_color, epsilon, Q, empty=0):
    # decide if we play randomly (epsilon) or following the value function (1-epsilon)
    if random() < epsilon:
        # the agent play randomly
        agent_move_row, agent_move_column = random_move(board, agent_color, empty)
        return agent_move_row, agent_move_column
    else:
        # the agent makes his move based on the value function
        possible_states = states_that_can_be_reached_from(board, agent_color)
        value_of_state = nn.Q_eval(Q, possible_states)
        # value_of_state = [nn.Q_eval(Q, possible_states[i]) for i in range(len(possible_states))]
        max_index = np.argmax(value_of_state)
        chosen_state = possible_states[max_index]

        # old version that consider also the possibility that two states have the same value Q(state)

        # max_value = -np.inf
        # possible_choices = []
        # for state in possible_states:
        #    value_of_state = nn.Q_eval(Q, state)
        #
        #   if value_of_state > max_value:
        #        del possible_choices[:]
        #        possible_choices.append(state)
        #        max_value = value_of_state

        # this elif is actually useless because the confrontation between float can't be equal
        # (this is how it works in C++)
        #    elif value_of_state == max_value:
        #        possible_choices.append(state)

        # chosen_state = possible_choices[randint(0, len(possible_choices) - 1)]
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
    board[row_move][column_move] = color_of_player
    return row_move, column_move


def get_last_occupied_row_in_column(board, column, empty=0):
    # !!Warning: it returns len(board) if the column is empty !!
    # !!Warning len(board) is out of bound!!

    row = len(board)
    while row > 0 and board[row - 1][column] != empty:
        row = row - 1
    return row


def is_full(board, empty=0):
    # np.count_nonzero(board[0] == 0) it seems counterintuitive but this line actually counts how many zeros
    # there are in board[0]
    if np.count_nonzero(board[0] == empty) == 0:
        return True
    return False


def is_winning(board, last_move_column, last_move_row=-2, empty=0, red=-1, yellow=1):
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


def plot_performances(total_games_played, probability_of_success):
    fig, ax = plt.subplots()
    ax.set_ylim(([0, 1.09]))
    ax.plot(total_games_played, probability_of_success)
    ax.set_xlabel("games played")
    ax.set_ylabel("probability of success", labelpad=0.1)
    fig.show()


def compute_target_y(Q, SA, r, S_prime, agent_color=1, gamma=1):
    # if SA[i] is a terminal state
    if r != 0:
        y_target_state = r
    else:
        possible_interstate_from_S_prime = states_that_can_be_reached_from(S_prime, agent_color)
        Q_of_possible_states = nn.Q_eval(Q, possible_interstate_from_S_prime)
        # Q_of_possible_states = [nn.Q_eval(Q, interstate) for interstate in possible_interstate_from_S_prime]
        y_target_state = r + gamma * np.max(Q_of_possible_states)
    return y_target_state


def select_the_batch_and_train_the_NN(batch_size, Q, SA_intermediate_state, r, S_prime, agent_color=1):
    my_batch = [randrange(len(r)) for _ in range(batch_size)]
    selected_SA_intermediate_state = [SA_intermediate_state[i] for i in my_batch]
    selected_r = [r[i] for i in my_batch]
    selected_S_prime = [S_prime[i] for i in my_batch]

    nn.train_my_NN(Q, selected_SA_intermediate_state, selected_r, selected_S_prime, agent_color)
