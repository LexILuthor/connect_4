import numpy as np
#----------------------------------
import play_move_functions as play
import copy
#----------------------------------
# The following function creates a full memory
# of (random) experiences. 
def create_memory_player1(
    memory_size,
    n_rows, 
    n_columns, 
    Q_agent, 
    Q_environment, 
    rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0], 
    epsilon=1):

	S = np.zeros([n_rows, n_columns]).astype(int)
	memory = []
	for i in range(memory_size):
		(S, a, r, S_prime) = copy.deepcopy(play.play_move_vs_AI_environment(
        Q_agent,
        Q_environment,
        S,
        rewards_Wi_Lo_Dr_De,
        is_agent_player1 = True,
        agent_color=1, 
        ambient_color=-1, 
        epsilon_agent=0.1,
        epsilon_environment = 0.1,
        empty = 0))        
		memory.append((S, a, r, S_prime))
	return memory


def create_memory_player2(
    memory_size,
    n_rows, 
    n_columns, 
    Q_agent, 
    Q_environment, 
    rewards_Wi_Lo_Dr_De=[10, -10, -0.1, 0], 
    epsilon=1):

	S = np.zeros([n_rows, n_columns]).astype(int)
	memory = []
	for i in range(memory_size):
		(S, a, r, S_prime) = copy.deepcopy(play.play_move_vs_AI_environment(
        Q_agent,
        Q_environment,
        S,
        rewards_Wi_Lo_Dr_De,
        is_agent_player1 = False,
        agent_color=1, 
        ambient_color=-1, 
        epsilon_agent=0.1,
        epsilon_environment = 0.1,
        empty = 0))        
		memory.append((S, a, r, S_prime))
	return memory
