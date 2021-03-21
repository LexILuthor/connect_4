# To do: make a program to allow users to play against the AI
import os
import neural_network as nn
import numpy as np

# Load AI
# Initialize NN
Q = nn.create_NN(7,8)
# Restore the weights
Q.load_weights('./weights.h5')
# note: even answers are positive, odd ones are negative
answers = ["y", "n", "Y", "N", "yes", "no", "Yes", "No"]
wanna_play = True



board = np.zeros([7,8]).astype(int)



def wrapper(wanna_play):
	while wanna_play == True:
		os.system("clear")
		print("Do you want to play a game of connect 4? (y/n)")
		answ = input()
		while answ not in answers:
			print("\nSorry, I can't understand your answer. Please, eneter a valid answer. (y/n)")
			answ = input()
		if answers.index(answ) % 2 == 0:
			print("Great! Let's play!")
			print("The AI plays first:")
			np.argmax(Q_eval(Q, board))
			# DA QUIIIII DA FINIRE!! ++++++++++++++++++++++++++++++++++++++++++++++++++


		else:
			print("Too bad. See you next time!")
			wanna_play = False




#----------------------
# graphic


wrapper(wanna_play)