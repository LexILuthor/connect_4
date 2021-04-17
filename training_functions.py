import numpy as np
import random
import copy
import os

import play_move_functions as play
import secondary_Functions as secFun
import neural_network as nn 
import memory as mem


import time 


# In this training algorithm the target is computed by using a copy of the NN
# which is frozen in time and it is updated every "freeze_steps"
def freeze_train_vs_random(
	Q,
	n_rows,
	n_columns, 
	memory,
	number_of_moves = 10000,
	freeze_steps = 100,
	rewards_Wi_Lo_Dr_De = [10, -10, -0.1, 0],
	train_freq = 2,
	batch_size = 10,
	gamma = 0.9,
	epsilon = 0.1,
	n_epochs = 2,
	):
	# initialize nn for computing target
    Q_target = copy.copy(Q)
    for move in range(number_of_moves):
        S = np.zeros([n_rows, n_columns]).astype(int)
        # Make one turn (agent plays a move, environment plays its move)
        (S, a, r, S_prime) = copy.deepcopy(play.play_move(Q, S, [10, -10, -0.1, 0], epsilon=epsilon))
#-----------------------------------------------------
        # delete random element
        memory.pop(random.randrange(len(memory)))
        # add new memory
        memory.append((S, a, r, S_prime))
        # S_prime is the next state S
        S = np.copy(S_prime)
#-----------------------------------------------------
		# freeze
        if (move % freeze_steps == 0):
            Q_target = copy.copy(Q)
#       TRAINING
        if (move > batch_size) and (move % train_freq == 0):
            batch = random.sample(memory, batch_size)
            nn.train_my_NN(Q, Q_target, batch, gamma, n_epochs)
    return Q


# Training algorithm where 2 AI (Q_player1 and Q_player2) are trained simultaneusly. 
# For both of them we freeze the NN that computes the target for "freeze_steps_player1"
# and "freeze_steps_player2" respectively. 
# Observe that there is the option of making the training "episodic". If "epsiodic == False",
# then the training is continuous. Namely when the agents are in a terminal state, they are 
# bounced back to the initial state and nothing special happens. 
# When "epsiodic == True", then whenever one agent reaches a terminal state (i.e. the end of
# an episode) then the memory is resetted and refilled using the current NN and the given 
# epsilon for that agent.

def freeze_train_player1(
	Q_player1,
	Q_player2,
	n_rows,
	n_columns, 
	memory_player1,
	episodic = False,
	number_of_moves = 10000,
	freeze_steps_player1 = 500,
	train_freq_player1 = 2,
	batch_size_player1 = 10,
	rewards_Wi_Lo_Dr_De = [10, -10, -0.1, 0],
	epsilon_player1 = 0.1,
	gamma = 0.9,
	n_epochs = 2,
	color_player1 = 1,
	color_player2 = -1
	):
	# initialize nn for computing target for both players
    Q_target_player1 = copy.copy(Q_player1)
#--------------------------------------------------------------------------------
    	# PLAYER 1
    S_player1 = np.zeros([n_rows, n_columns]).astype(int)
    for move in range(number_of_moves):
        os.system("clear")

        #  # for debugging
        #print(S_player1)
        #time.sleep(2)

    	# print move number and percentage
        perc = round((move/number_of_moves)*100, 2)
        print("\nStep ", move, ". Training is ", perc, " % complete. \n", sep="")
        # Make one turn (agent plays a move, environment plays its move)
        (S_player1, a_player1, r_player1, S_prime_player1) = copy.deepcopy(
            play.play_move_vs_AI_environment(
                Q_agent = Q_player1,
                Q_environment = Q_player2,
                S = S_player1,
                rewards_Wi_Lo_Dr_De = rewards_Wi_Lo_Dr_De,
                is_agent_player1 = True,
                agent_color = color_player1,
                ambient_color = color_player2,
                epsilon_agent = epsilon_player1,
                epsilon_environment = 0,
                empty = 0
                ))
        # I save this for adding to batch **************************************** TO CLEAN UP
        t1 = copy.copy((S_player1, a_player1, r_player1, S_prime_player1))
        # delete random element from memory of player 1
        memory_player1.pop(random.randrange(len(memory_player1)))
        # add new memory
        memory_player1.append((S_player1, a_player1, r_player1, S_prime_player1))
        # S_prime is the next state S
        S_player1 = np.copy(S_prime_player1)
		# freeze
        if (move % freeze_steps_player1 == 0):
            Q_target_player1 = copy.copy(Q_player1)
        # training
        if (move % train_freq_player1 == 0):
            batch_player1 = random.sample(memory_player1, batch_size_player1)
            # ********************************************************** CER (zhang burton) TO clean up
            batch_player1.append(t1) 
            nn.train_my_NN(Q_player1, Q_target_player1, batch_player1, gamma, n_epochs)
        # check whether the parameter "episodic" is True or False
        if episodic == True:
            # check whether it is the end of the episode  
            if r_player1 != 0:
                memory_size_player1 = len(memory_player1)
                memory_player1 = mem.create_memory(
                    memory_size_player1,
                    n_rows, 
                    n_columns, 
                    Q_player1, 
                    epsilon = epsilon_player1
                    )
    return Q_player1


def freeze_train_player2(
	Q_player1,
	Q_player2,
	n_rows,
	n_columns, 
	memory_player2,
	episodic = False,
	number_of_moves = 50000,
	freeze_steps_player2 = 100,
	train_freq_player2 = 2,
	batch_size_player2 = 10,
	rewards_Wi_Lo_Dr_De = [10, -10, -0.1, 0],
	epsilon_player2 = 0.1,
	gamma = 0.9,
	n_epochs = 2,
	color_player1 = 1,
	color_player2 = -1
	):
	# initialize nn for computing target for both players
    Q_target_player2 = copy.copy(Q_player2)
#----------------------------------------------------------------------------------
		# PLAYER 2
	# Initialize S for player 2
    board = np.zeros([n_rows, n_columns]).astype(int)
    (first_move_row, first_move_col) = secFun.agent_move_following_epsilon_Q(
        board = board, 
        agent_color = color_player1,
        epsilon = 0, 
        Q = Q_player1
        )
    board[first_move_row, first_move_col] = color_player1
    S_player2 = copy.copy(board)
    for move in range(number_of_moves):
        os.system("clear")

        ## for debugging
        #print(S_player2)
        #time.sleep(2)

    	# print move number and percentage
        perc = round((move/number_of_moves)*100, 2)
        print("\nStep ", move, ". Training is ", perc, " % complete. \n", sep="")
        # Make one turn (environment plays a move, agent plays its move)
        (S_player2, a_player2, r_player2, S_prime_player2) = copy.deepcopy(
            play.play_move_vs_AI_environment(
                Q_agent = Q_player2,
                Q_environment = Q_player1,
                S = S_player2,
                rewards_Wi_Lo_Dr_De = rewards_Wi_Lo_Dr_De,
                is_agent_player1 = False,
                agent_color = color_player2,
                ambient_color = color_player1,
                epsilon_agent = epsilon_player2,
                epsilon_environment = 0,
                empty = 0
                ))
        # I save this for adding to batch **************************************** TO CLEAN UP
        t2 = copy.copy((S_player2, a_player2, r_player2, S_prime_player2))
        # delete random element from memory of player 2
        memory_player2.pop(random.randrange(len(memory_player2)))
        # add new memory
        memory_player2.append((S_player2, a_player2, r_player2, S_prime_player2))
        # S_prime is the next state S
        S_player2 = np.copy(S_prime_player2)
		# freeze
        if (move % freeze_steps_player2 == 0):
            Q_target_player2 = copy.copy(Q_player2)
        # training player 2
        if (move % train_freq_player2 == 0):
            batch_player2 = random.sample(memory_player2, batch_size_player2)
            # ********************************************************** CER (zhang burton) TO clean up
            batch_player2.append(t2) 
            nn.train_my_NN(Q_player2, Q_target_player2, batch_player2, gamma, n_epochs)
        # check whether the parameter "episodic" is True or False
        if episodic == True:
            # check whether it is the end of the episode  
            if r_player2 != 0:
                memory_size_player2 = len(memory_player2)
                memory_player2 = mem.create_memory(
                    memory_size_player2,
                    n_rows, 
                    n_columns, 
                    Q_player2, 
                    epsilon = epsilon_player2
                    )
    return Q_player2