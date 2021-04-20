class Agent:
    """docstring for ClassName"""
    def __init__(self, name, which_player, Q, epsilon):
        self.name = name                                 # name of the agent
        self.which_player = which_player                 # 1 or 2
        self.Q = Q                                       # Q is the neural network
        self.epsilon = epsilon                           # for the epsilon-greedy strategy during training


class Board(object):
    """docstring for Board"""
    def __init__(self, board_matrix, n_rows, n_columns, color_player1 = 1, color_player2 = -1):
        self.board = board_matrix
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.color_player1 = color_player1
        self.color_player2 = color_player2

        def print(color_player1 = 1, color_player2 = -1, empty = 0):
            # !!! TO DO !!!
            # write a function to print the board
		