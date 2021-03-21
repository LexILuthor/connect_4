import os
import numpy as np
import secondary_Functions as secFun
import myFunctions as myFun
import neural_network as nn
import copy
import random


def main():
    # Initialize NN
    Q = nn.create_NN(7,8)
    


    train_freq = 50
    batch_size = 20
    gamma = 0.9
    epsilon = 0.05
    learn_rate=0.01
    n_epochs = 2


    rewards_Wi_Lo_Dr_De = [10, -10, -0.1, 0]
    number_of_moves = 20000
    memory_size = 10000  # Let's set the memory capacity
 
    # Initialize memory
    memory = []
    # Create empty board
    board = np.zeros([7, 8]).astype(int)
    # Initialize NN
 
    Q = nn.create_NN(7, 8)
    # Initialize first state
    S = np.copy(board)

    # FOR DEBUGGING
    count_win = 0
    count_lose = 0
    train_freq = 10
    batch_size = 5

    for move in range(number_of_moves):
        # Make one turn (agent plays a move, environment plays its move)
        (S, a, r, S_prime) = copy.deepcopy(myFun.play_move(Q, S, rewards_Wi_Lo_Dr_De, epsilon=epsilon))        
#-----------------------------------------------------
#           UPDATE MEMORY

        # if memory is still no full
        if len(memory) < memory_size:
            # add experience to memory
            memory.append((S, a, r, S_prime))
        # otherwise we need to empty a slot
        else:
            # delete first element
            memory.pop(0)
            # add new memory

            memory.append((S, a, r, S_prime))
        # S_prime is the next state S
        S = np.copy(S_prime)
#----------------------------------------------------
#           TRAINING

        if (move > batch_size) and (move % train_freq == 0):
       
            batch = random.sample(memory, batch_size)
            nn.train_my_NN(Q, batch, gamma, n_epochs)

#-----------------------------------------------------
#   THIS IS A SUPER WEIRD STEP... APPARENTLY I NEED TO TRAIN 
#   THE NN A LITTLE BIT BEFORE LOADING THE OLD WEIGHTS \('.')/
        if move == 3*batch_size:
            # Restore the weights
            Q.load_weights('./weights.h5')
# #------------------------------------------------------
# #           DYNAMIC EPSILON

#         # Change epsilon after 1K moves
#         if move > 5000:
#             epsilon = 0.1
#         if move > 10000:
#             epsilon = 0.05
#------------------------------------------------------- 

#---------- END LEARNING LOOP -------------------------


#-------------------------------------------------------   
#      SAVE WEIGHTS
    Q.save_weights('./weights.h5')

#------------------------------------------------------
#          TEST 
    

    count_win = 0
    count_lose = 0
    count_draw = 0
    for move in range(number_of_moves):
        S, a, r, S_prime = copy.deepcopy(myFun.play_move(Q, S, rewards_Wi_Lo_Dr_De, epsilon=0))
        if r == rewards_Wi_Lo_Dr_De[0]:
            count_win += 1
        if r == rewards_Wi_Lo_Dr_De[1]:
            count_lose += 1
        if r == rewards_Wi_Lo_Dr_De[2]:
            count_draw +=1
        # S_prime is the next state S
        S = np.copy(S_prime)

    print("number of moves:")
    print(number_of_moves)
    print("training frequency:")
    print(train_freq)
    print("\nNumber games won:")
    print(count_win)
    print("Number games lost:")
    print(count_lose)
    print("Number of games ended in a draw:")
    print(count_draw)
    num_games = count_win + count_lose + count_draw
    print("The AI won the ", round((count_win/(num_games))*100,2), "% ", "of the games." )
    
#------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
