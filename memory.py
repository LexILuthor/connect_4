import numpy as np
import os
import time 
#----------------------------------
import play_move_functions as play
import secondary_Functions as secFun
import copy
#----------------------------------
# The following function creates a full memory
# of (random) experiences. 
def create_memory_player1(
    memory_size,
    n_rows, 
    n_columns, 
    Q_agent, 
    Q_environment, 
    rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0], 
    epsilon=1):

    S = copy.deepcopy(np.zeros([n_rows, n_columns]).astype(int))
    memory = []
    for i in range(memory_size):
        (S, a, r, S_prime) = copy.deepcopy(play.play_move_vs_AI_environment(
        Q_agent,
        Q_environment,
        S,
        rewards_Wi_Lo_Dr_De,
        is_agent_player1 = True,
        agent_color=1, 
        ambient_color=-1, 
        epsilon_agent=epsilon,
        epsilon_environment = 0,
        empty = 0))        
        memory.append((S, a, r, S_prime))
        S = copy.deepcopy(S_prime)
    return memory


def create_memory_player2(
    memory_size,
    n_rows, 
    n_columns, 
    Q_agent, 
    Q_environment, 
    rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0], 
    epsilon=1):

    board = np.zeros([n_rows, n_columns]).astype(int)
    (first_move_row, first_move_col) = secFun.agent_move_following_epsilon_Q(
        board = board, 
        agent_color = 1,
        epsilon = 0, 
        Q = Q_environment
        )
    board[first_move_row, first_move_col] = 1 # !!!! WARNING !!! This is not a good solution
    S = copy.deepcopy(board)
    # debugging
    #print(S)
    #time.sleep(3)
    memory = []
    for i in range(memory_size):
        (S, a, r, S_prime) = copy.deepcopy(play.play_move_vs_AI_environment(
        Q_agent,
        Q_environment,
        S,
        rewards_Wi_Lo_Dr_De,
        is_agent_player1 = False,
        agent_color=-1, 
        ambient_color=1, 
        epsilon_agent=epsilon,
        epsilon_environment = 0,
        empty = 0))        
        memory.append((S, a, r, S_prime))
        S = copy.deepcopy(S_prime)

    # for debugging
    os.system("clear")
    print(S)
    time.sleep(1)
    return memory
