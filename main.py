import numpy as np
import secondary_Functions as secFun
import myFunctions as myFun
import neural_network as nn
import copy

def main():
    number_of_moves = 100
    memory_size = 10  # Let's set the memory capacity
    # Initialize memory
    memory = []
    # Create empty board
    board = np.zeros([5,6]).astype(int)
    # Initialize NN
    Q = nn.create_NN(5,6)
    # Initialize first state
    S = np.copy(board)
    
    # let's fill up the memory first
    for move in range(number_of_moves):
        (S, r, a, S_prime) = copy.deepcopy(myFun.play_move(Q, S, epsilon=0.2))
        # if memory is still no full
        if len(memory) < memory_size:
            # add experience to memory
            memory.append((S,r,a,S_prime))
        # otherwise we need to empty a slot
        else:
            # delete first element
            memory.pop(0)
            # add new memory
            memory.append((S,r,a,S_prime))

        # S_prime is the next state S
        S = np.copy(S_prime)



    print(memory)
    print(memory[3])
    print(memory[4])

if __name__ == '__main__':
    main()
