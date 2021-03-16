import numpy as np
import json

import myFunctions as myFun


def testingmyfunction():
    board = np.array(
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0],
         [1, 1, 1, 1, -1, -1]
         ], dtype=np.int8)

    training_steps = 10
    Q = {}

    # Memory of the agent------------------------------------
    memory_size = 100       # Let's set the memory capacity
    D = []                  # Initialize the memory (as a list)

    # Memory of the agent------------------------------------

    for i in range(training_steps):
        SA_intermediate_state, r, S_prime = myFun.play_a_game(Q)
        new_experience = [SA_intermediate_state, r, S_prime]        # This is the new experience to be added in the memory D

        if len(D) >= memory_size:       # Check if the memory is already full
            D.pop(0)                    # we remove the first element

        D.append(new_experience)        # and then we add the         
        



if __name__ == '__main__':
    testingmyfunction()
