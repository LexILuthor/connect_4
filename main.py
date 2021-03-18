import numpy as np
import secondary_Functions as secFun
import myFunctions as myFun
import neural_network as nn


def main():
    number_of_moves = 1000
    memory_size = 100  # Let's set the memory capacity

    # TO DO: FARE UN LOOP PER RIEMPIRE LA MEMORIA E VEDERE SE FUNZIONA

    # Simple example for debugging
    Q = nn.create_NN(3,4)
    agent_color = 1
    epsilon = 0.3
    board = np.zeros([3,4]).astype(int)
    board[2,:] = np.array([-1, 1, -1, 1]).astype(int)


    #print(secFun.agent_move_following_epsilon_Q(board, agent_color, epsilon, Q, 0))
    S = np.copy(board)
    S_prime = np.copy(myFun.play_move(Q, S))[3]
    print(myFun.play_move(Q, S))
    S = np.copy(S_prime)
    print(myFun.play_move(Q, S))

if __name__ == '__main__':
    main()
