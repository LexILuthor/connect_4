import numpy as np
#----------------------------------
import play_move_functions as play
import copy
#----------------------------------
# The following function creates a full memory
# of (random) experiences. 
def create_memory(memory_size, n_rows, n_columns, Q, rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0], epsilon=1):
	S = np.zeros([n_rows, n_columns]).astype(int)
	memory = []
	for i in range(memory_size):
		(S, a, r, S_prime) = copy.deepcopy(play.play_move(Q, S, rewards_Wi_Lo_Dr_De, epsilon=epsilon))        
		memory.append((S, a, r, S_prime))
	return memory

# CER = Combined Experience Replay 
# (see "A deeper look at experience replay" by Zhang & Burton, 2017)
