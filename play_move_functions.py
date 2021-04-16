import numpy as np
import random
import copy 

import secondary_Functions as secFun


# ----------------------------------------------------------------------------------------------------------
# Function to make the agent play only one move and returns a 4-tuple (s,a,r,s') of experience
# If it ends in a terminal state, then S' is set to the empty board. 
#
def play_move(Q, S, rewards_Wi_Lo_Dr_De, agent_color=1, ambient_color=-1, epsilon=0.1, empty = 0):
    # agent makes a move
    agent_move_row, agent_move_column = copy.deepcopy(secFun.agent_move_following_epsilon_Q(S, agent_color, epsilon, Q, empty))
    current_state = copy.deepcopy(S)
    a = copy.copy(agent_move_column)
    # define the intermidiate state
    inter_state = copy.deepcopy(S)
    inter_state[agent_move_row, agent_move_column] = copy.copy(agent_color)
    # check if the agent won
    if secFun.is_winning(inter_state, agent_move_row, agent_move_column, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[0])
        return current_state, a, r, S_prime
    # check if board is full then it is a draw
    if secFun.is_full(inter_state, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[2])
        return current_state, a, r, S_prime
    # ambient makes a (random) move
    ambient_move_row, ambient_move_column = copy.deepcopy(secFun.ambient_move(inter_state, ambient_color, empty))
    S_prime = copy.deepcopy(inter_state)
    S_prime[ambient_move_row, ambient_move_column] = copy.copy(ambient_color)
    # check if ambient won
    if secFun.is_winning(S_prime, ambient_move_row, ambient_move_column, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[1])
        return current_state, a, r, S_prime
    # check if board is full
    if secFun.is_full(S_prime, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[2])
        return current_state, a, r, S_prime
    # it was a "nothing happens" action
    else:
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[3])
        return current_state, a, r, S_prime

# ----------------------------------------------------------------------------------------------------------
# FOR PLAYER 2
# Function to make the agent play only one move and returns a 4-tuple (s,a,r,s') of experience
# If it ends in a terminal state, then S' is set to the empty board. 
#
def play_move_vs_AI_environment(
    Q_agent,
    Q_environment,
    S,
    rewards_Wi_Lo_Dr_De,
    is_agent_player1 = True,
    agent_color=1, 
    ambient_color=-1, 
    epsilon_agent=0.1,
    epsilon_environment = 0.1,
    empty = 0):

    # agent makes a move
    agent_move_row, agent_move_column = copy.deepcopy(secFun.agent_move_following_epsilon_Q(S, agent_color, epsilon_agent, Q_agent, empty))
    current_state = copy.deepcopy(S)
    a = copy.copy(agent_move_column)
    # define the intermidiate state
    inter_state = copy.deepcopy(S)
    inter_state[agent_move_row, agent_move_column] = copy.copy(agent_color)
    # check if the agent won
    if secFun.is_winning(inter_state, agent_move_row, agent_move_column, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[0])
        return current_state, a, r, S_prime
    
    # check if board is full then it is a draw
    if secFun.is_full(inter_state, empty):
        # Since we are in a terminal state S_prime is set to be the empty board so the game start again
        S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[2])
        return current_state, a, r, S_prime

    # ambient makes a  move
    ambient_move_row, ambient_move_column = copy.deepcopy(secFun.AI_environment(inter_state, Q_environment, epsilon_environment, ambient_color, empty))
    S_prime = copy.deepcopy(inter_state)
    S_prime[ambient_move_row, ambient_move_column] = copy.copy(ambient_color)

    # check if ambient won
    if secFun.is_winning(S_prime, ambient_move_row, ambient_move_column, empty):
        # Since we are in a terminal state,
        # therefore S_prime is set to be the initial state and the game start again
        if is_agent_player1 == True:
            S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        else:
            board = np.zeros(np.shape(S)).astype(int)
            (first_move_row, first_move_col) = secFun.agent_move_following_epsilon_Q(
                board = board, 
                agent_color = ambient_color,
                epsilon = epsilon_environment, 
                Q = Q_environment
            )
            board[first_move_row, first_move_col] = ambient_color
            S_prime = copy.deepcopy(board)

        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[1])
        return current_state, a, r, S_prime

    # check if board is full
    if secFun.is_full(S_prime, empty):
        # Since we are in a terminal state,
        # therefore S_prime is set to be the initial state and the game start again
        if is_agent_player1 == True:
            S_prime = copy.deepcopy(np.zeros(np.shape(S)).astype(int))
        else:
            board = np.zeros(np.shape(S)).astype(int)
            (first_move_row, first_move_col) = secFun.agent_move_following_epsilon_Q(
                board = board, 
                agent_color = ambient_color,
                epsilon = epsilon_environment, 
                Q = Q_environment
            )
            board[first_move_row, first_move_col] = ambient_color
            S_prime = copy.deepcopy(board)

        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[1])
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[2])
        return current_state, a, r, S_prime

    # it was a "nothing happens" action
    else:
        r = copy.deepcopy(rewards_Wi_Lo_Dr_De[3])
        return current_state, a, r, S_prime