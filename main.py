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
         ])

    myFun.print_board(board)
    myFun.play_a_game([])


if __name__ == '__main__':
    testingmyfunction()
