# To do: make a program to allow users to play against the AI
import time 
import os
import numpy as np
import secondary_Functions as secFun
import neural_network as nn

# note: even answers are positive, odd ones are negative
answers = ["y", "n", "Y", "N", "yes", "no", "Yes", "No"]
wanna_play = True



# Initialize AI

Q_agent = nn.load_NN("cnn_sigmoid_pl1", 6,7)
Q_environment = nn.load_NN("test", 6,7)


def wrapper(wanna_play, agent_color = 1, human_color = -1, n_actions = 7):
	while wanna_play == True:
		os.system("clear")
		print("Do you want to play a game of connect 4? (y/n)")
		answ = input()
		while answ not in answers:
			print("\nSorry, I can't understand your answer. Please, eneter a valid answer. (y/n)")
			answ = input()
		if answers.index(answ) % 2 == 0:
			print("Great! Let's play!")
			os.system("clear")
			# Initialize board
			board = np.zeros([6,7]).astype(int)
			is_game_ended = False
			while is_game_ended == False:
				# AI plays
				agent_move_row, agent_move_column = secFun.agent_move_following_epsilon_Q(board, agent_color, 0, Q_agent, 0)
				print("The AI plays: ", agent_move_column)
				board[agent_move_row, agent_move_column] = agent_color
				print(board)
				# Check if AI wins
				if secFun.is_winning(board, agent_move_row, agent_move_column) == True:
					print("The AI wins!")
					time.sleep(3)
					break					
				time.sleep(1)
				# Human plays
				human_move_row, human_move_column = secFun.human_move(board) #AI_environment(board, Q_environment)
				board[human_move_row, human_move_column] = human_color
				# Check if human wins
				if secFun.is_winning(board, human_move_row, human_move_column) == True:
					print("Human wins!")
					time.sleep(3)
					break
					#is_game_ended = True
				time.sleep(1)

				# Check if board is full
				if secFun.is_full(board) == True:
					print("The game ended in a draw!")
					time.sleep(3)
					break


		else:
			print("Too bad. See you next time!")
			wanna_play = False




#----------------------
# graphic


wrapper(wanna_play)