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
    value_function = {}


    for i in range(training_steps):
        print(i)
        states_reached, result = myFun.play_a_game(value_function)
        myFun.update_value_function(value_function, states_reached, result)






if __name__ == '__main__':
    testingmyfunction()
