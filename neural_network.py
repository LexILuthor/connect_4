# !!!! WARNING !!!
# Here I just copy-pasted a CNN written for MNIST as personal exercise
# I am currently in the process of making it useful for connect 4
# !!! THIS IS WORK IN PROGRESS !!!
import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers,  losses
import tensorflow.keras.backend as kb
import copy


def create_NN(n_rows, n_columns):
    Q = tf.keras.Sequential([
        layers.Conv2D(10, (3, 3), activation='relu', input_shape=(n_rows, n_columns, 1)),
        layers.Dropout(0.1),  # This is for regularization
        layers.Flatten(),
        layers.Dense(20, activation='relu'),  # This can be changed later
        layers.Dropout(0.2),
        layers.Dense(n_columns)  # The number of actions is equal to the number of columns
    ])
    Q.compile(optimizer='SGD',
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])
    return Q



# Q_eval is a function that given our neural network Q and the current state s returns the values of all actions
def Q_eval(Q, current_state):
    # we need to reshape the state into a tensor
    current_state = np.array(current_state)
    (n_rows, n_columns) = np.shape(current_state)
    current_state = current_state.reshape(1, n_rows, n_columns, 1)
    #return Q.predict(result)[0]
    return np.array(Q(current_state)[0])



# Function that creates target from experience
def create_target(Q, experience, gamma):
	len_experience = len(experience)
	target = []
	for exp in range(len_experience):
		a = experience[exp][1]
		S = copy.deepcopy(experience[exp][0])
		# if reward is not zero then we end up in a terminal state
		if experience[exp][2] == 0:
			actual_target = gamma * np.max(Q_eval(Q, experience[exp][3]))
		else:
			actual_target = experience[exp][2]
		vector = copy.deepcopy(Q_eval(Q, S))
		vector[a] = actual_target
		target.append(vector)
	target = np.array(target)
	return target

# Extract S to train
def create_x_train(experience):
	len_experience = len(experience)
	n_rows = np.shape(experience[0][0])[0]
	n_columns = np.shape(experience[0][0])[1]

	S = []
	for exp in range(len_experience):
		S.append(experience[exp][0])
	S = np.array(S)
	S = S.reshape(len_experience,n_rows, n_columns, 1)
	return S



# a function that given a batch of training set coming from memory D, and the discount rate gamma
# computes the target value, and then trains the neural network 
# !!! WARNING !!!
# I expect experience to be a list with N elements and each element is a 4-list
def train_my_NN(Q, experience, gamma):
	len_experience = len(experience)
	target = create_target(Q, experience, gamma)
	x_train = create_x_train(experience)
	history = Q.fit(x_train, target, batch_size=len_experience, epochs = 1)
	#print(history.history)
	return
		
   
