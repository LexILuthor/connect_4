import copy
import numpy as np

import play_move_functions as play
import secondary_Functions as secFun

#----------------------------------------------------------------------------------
#    This function test performance of an AI agent as PLAYER 1 vs random environment
def test_vs_random(
	Q,
	n_rows,
	n_columns,
	number_of_moves = 1000,
	rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0],
	):
    S = np.zeros([n_rows,n_columns]).astype(int)
    count_win = 0
    count_lose = 0
    count_draw = 0
    for move in range(number_of_moves):
        S, a, r, S_prime = copy.deepcopy(play.play_move(Q, S, rewards_Wi_Lo_Dr_De, epsilon = 0, empty=0))
        if r == rewards_Wi_Lo_Dr_De[0]:
            count_win += 1
        if r == rewards_Wi_Lo_Dr_De[1]:
            count_lose += 1
        if r == rewards_Wi_Lo_Dr_De[2]:
            count_draw +=1
        # S_prime is the next state S
        S = np.copy(S_prime)

    print("number of moves:")
    print(number_of_moves)
    #print("training frequency:")
    #print(train_freq)
    print("\nNumber games won:")
    print(count_win)
    print("Number games lost:")
    print(count_lose)
    print("Number of games ended in a draw:")
    print(count_draw)
    num_games = count_win + count_lose + count_draw
    success_rate = (count_win/(num_games))*100
    print("The AI won the ", round(success_rate,2), "% ", "of the games vs a random player." )
    return success_rate


#----------------------------------------------------------------------------------
#    This function test performance of an AI agent as PLAYER 1 vs AI environment
def test_vs_AI_player1(
    Q_agent,
    Q_environment,
    n_rows,
    n_columns,
    number_of_moves = 1000,
    rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0],
    ):
    S = np.zeros([n_rows,n_columns]).astype(int)
    count_win = 0
    count_lose = 0
    count_draw = 0
    for move in range(number_of_moves):
        S, a, r, S_prime = copy.deepcopy(play.play_move_vs_AI_environment(
            Q_agent,
            Q_environment, 
            S, 
            rewards_Wi_Lo_Dr_De, 
            epsilon_agent = 0, 
            epsilon_environment = 0, 
            empty=0
            ))
        if r == rewards_Wi_Lo_Dr_De[0]:
            count_win += 1
        if r == rewards_Wi_Lo_Dr_De[1]:
            count_lose += 1
        if r == rewards_Wi_Lo_Dr_De[2]:
            count_draw +=1
        # S_prime is the next state S
        S = np.copy(S_prime)

    print("number of moves:")
    print(number_of_moves)
    #print("training frequency:")
    #print(train_freq)
    print("\nNumber games won:")
    print(count_win)
    print("Number games lost:")
    print(count_lose)
    print("Number of games ended in a draw:")
    print(count_draw)
    num_games = count_win + count_lose + count_draw
    success_rate = (count_win/(num_games))*100
    print("The AI player 1 won the ", round(success_rate,2), "% ", "of the games." )
    return success_rate


#----------------------------------------------------------------------------------
#    This function test performance of an AI agent as PLAYER 2 vs AI environment
def test_vs_AI_player2(
    Q_environment,
    Q_agent,
    n_rows,
    n_columns,
    number_of_moves = 1000,
    rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0],
    ):
    S = np.zeros([n_rows,n_columns]).astype(int)
    # Initialize S for player 2
    board = np.zeros([n_rows, n_columns]).astype(int)
    (first_move_row, first_move_col) = secFun.agent_move_following_epsilon_Q(
        board = board, 
        agent_color = 1,
        epsilon = 0, 
        Q = Q_environment
        )
    board[first_move_row, first_move_col] = 1
    S = copy.copy(board)
    count_win = 0
    count_lose = 0
    count_draw = 0
    for move in range(number_of_moves):
        S, a, r, S_prime = copy.deepcopy(play.play_move_vs_AI_environment(
            Q_agent,
            Q_environment, 
            S, 
            rewards_Wi_Lo_Dr_De,
            is_agent_player1 = False, 
            epsilon_agent = 0, 
            epsilon_environment = 0, 
            empty=0
            ))
        if r == rewards_Wi_Lo_Dr_De[0]:
            count_win += 1
        if r == rewards_Wi_Lo_Dr_De[1]:
            count_lose += 1
        if r == rewards_Wi_Lo_Dr_De[2]:
            count_draw +=1
        # S_prime is the next state S
        S = np.copy(S_prime)

    print("number of moves:")
    print(number_of_moves)
    #print("training frequency:")
    #print(train_freq)
    print("\nNumber games won:")
    print(count_win)
    print("Number games lost:")
    print(count_lose)
    print("Number of games ended in a draw:")
    print(count_draw)
    num_games = count_win + count_lose + count_draw
    success_rate = (count_win/(num_games))*100
    print("The AI player 2 won the ", round(success_rate,2), "% ", "of the games." )
    return success_rate