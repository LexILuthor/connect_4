import numpy as np
import matplotlib.pyplot as plt
import random

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


# ----------------------------------------------------------------------------------------------------------

# Function to make the agent play only one move and returns a 4-tuple (s,a,r,s') of experience
# If it ends in a terminal state, then S' is set to the empty board. 
#
def play_move(Q, S, agent_color=1, ambient_color=-1, epsilon=0.1, empty = 0):

    # agent makes a move
    agent_move_row, agent_move_column = secFun.agent_move_following_epsilon_Q(S, agent_color, epsilon, Q, empty)
    current_state = np.copy(S)
    # for debugging
    a = agent_move_column
    # define the intermidiate state
    inter_state = np.copy(S)
    inter_state[agent_move_row, agent_move_column] = agent_color
    # check if the agent won
    if secFun.is_winning(inter_state, agent_move_row, agent_move_column, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = np.zeros(np.shape(S))
        r = 1

    
    # check if board is full then it is a draw
    if secFun.is_full(inter_state, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = np.zeros(np.shape(S))
        r = -0.5
    

    # ambient makes a (random) move
    ambient_move_row, ambient_move_column = secFun.ambient_move(inter_state, ambient_color, empty)
    S_prime = np.copy(inter_state)
    S_prime[ambient_move_row, ambient_move_column] = ambient_color

    # check if ambient won
    if secFun.is_winning(S_prime, ambient_move_row, ambient_move_column, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = np.zeros(np.shape(S))
        r = -1    
        

    # check if board is full
    if secFun.is_full(S_prime, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = np.zeros(np.shape(S))
        r = -0.5
      

    # it was a "nothing happens" action
    else:
        r = 0

    return current_state, a, r, S_prime

        # --------------------------------------------------------------------------------------------------------------
        # here the turn ends (both the agent and the ambient have done their move)