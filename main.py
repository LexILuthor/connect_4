import numpy as np

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


    value_function = {}
    myFun.play_a_game(value_function)


if __name__ == '__main__':
    testingmyfunction()
