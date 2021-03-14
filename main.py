import numpy as np

import myFunctions as myFun


def testingmyfunction():
    matrix = np.array(
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0],
         [1, 1, 1, 1,-1, -1]
         ])

    last_move_column = 0
    result = myFun.is_winning(matrix, last_move_column, empty=0, red=-1, yellow=1)
    print("the result is " + str(result))


if __name__ == '__main__':
    testingmyfunction()
