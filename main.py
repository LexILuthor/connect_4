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

    training_steps = 10000
    Q = {}

    for i in range(training_steps):
        SA_intermediate_state, r, S_prime = myFun.play_a_game(Q)


if __name__ == '__main__':
    testingmyfunction()
