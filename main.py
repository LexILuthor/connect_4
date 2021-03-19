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
    # Restore the weights
    Q.load_weights('./weights')



    rewards_Wi_Lo_Dr_De = [100, -100, -30, 0]
    number_of_moves = 10000
    memory_size = 1000  # Let's set the memory capacity
    # Initialize memory
    memory = []
    # Create empty board
    board = np.zeros([7,8]).astype(int)
    # Initialize NN
    Q = nn.create_NN(7,8)
    # Initialize first state
    S = np.copy(board)
    
    # FOR DEBUGGING
    count_win = 0
    count_lose = 0
    train_freq = 10
    batch_size = 5

    # let's fill up the memory first
    for move in range(number_of_moves):
        (S, a, r, S_prime) = copy.deepcopy(myFun.play_move(Q, S, rewards_Wi_Lo_Dr_De, epsilon=0.2))
        if r == rewards_Wi_Lo_Dr_De[0]:
            count_win += 1
        if r == rewards_Wi_Lo_Dr_De[1]:
            count_lose -= 1
        # if memory is still no full
        if len(memory) < memory_size:
            # add experience to memory
            memory.append((S,a,r,S_prime))
        # otherwise we need to empty a slot
        else:
            # delete first element
            memory.pop(0)
            # add new memory
            memory.append((S,r,a,S_prime))
        # S_prime is the next state S
        S = np.copy(S_prime)

        # train step
        if move % train_freq == train_freq -1:
            batch = random.sample(memory, batch_size)
            nn.train_my_NN(Q, batch, 1)

    

    # Save the weights
    Q.save_weights('./weights')


#---------- TEST ---------------------
    
    count_win = 0
    count_lose = 0
    for move in range(number_of_moves):
        (S, a, r, S_prime) = copy.deepcopy(myFun.play_move(Q, S, rewards_Wi_Lo_Dr_De, epsilon=0))
        if r == rewards_Wi_Lo_Dr_De[0]:
            count_win += 1
        if r == rewards_Wi_Lo_Dr_De[1]:
            count_lose -= 1
        # if memory is still no full
        if len(memory) < memory_size:
            # add experience to memory
            memory.append((S,a,r,S_prime))
        # otherwise we need to empty a slot
        else:
            # delete first element
            memory.pop(0)
            # add new memory
            memory.append((S,r,a,S_prime))
        # S_prime is the next state S
        S = np.copy(S_prime)


    print("\nNumber games won:")
    print(count_win)
    print("Number games lost:")
    print(-count_lose)
    

if __name__ == '__main__':
    main()
