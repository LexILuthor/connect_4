# !!!! WARNING !!!
# Here I just copy-pasted a CNN written for MNIST as personal exercise
# I am currently in the process of making it useful for connect 4
# !!! THIS IS WORK IN PROGRESS !!!
import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers,  losses
import tensorflow.keras.backend as kb
import copy


# a function that returns our initialized neural network
def create_NN(n_rows, n_columns):
    Q = tf.keras.Sequential([
        layers.Conv2D(10, (3, 3), activation='relu', input_shape=(n_rows, n_columns, 1)),
        layers.Dropout(0.1),  # This is for regularization
        layers.Flatten(),
        layers.Dense(20, activation='relu'),  # This can be changed later
        layers.Dropout(0.2),
        layers.Dense(n_columns)  # The number of actions is equal to the number of columns
    ])
    # Now we set the training configuration
	Q.compile(optimizer='SGD', 			# adam is a type of stochastic gradient descent 
              loss=tf.keras.losses.MeanSquaredError(),					# the loss is our loss function 
              metrics=['accuracy'])
    return Q


# Q_eval is a function that given our neural network Q and the current state s returns the values of all actions
def Q_eval(Q, current_state):
    # we need to reshape the state into a tensor
    current_state = np.array(current_state)
    (n_rows, n_columns) = np.shape(current_state)
    current_state = current_state.reshape(1, n_rows, n_columns, 1)
    #return Q.predict(result)[0]
    return np.array(Q(current_state)[0])


# a function that given a batch of training set coming from memory D, and the discount rate gamma
# computes the target value, and then trains the neural network 
# !!! WARNING !!!
# I expect experience to be a list with N elements and each element is a 4-list
def train_my_NN(experience, Q, gamma):
	# lenght of experience batch
	len_batch = len(experience)
	n_actions = np.shape(experience[0][0])[1]
	n_rows = np.shape(experience[0][0])[0]
	# initialize list states
	train_states = np.zeros([n_rows, n_actions, len_batch])
	# initialize target
	y = np.zeros([len_batch, n_actions])
	# initialize prediction
	y_pred = np.zeros([len_batch, n_actions])
	for i in range(len_batch):
		# the first element in each 4-list is the state s
		s = copy.deepcopy(experience[i][0])
		# write it in the train states
		train_states[:,:, i] = copy.deepcopy(s)
		# the second element is the action a
		a = copy.deepcopy(experience[i][1])
		# let's compute the predicted value of (s,a)
		y_pred[i, :] = copy.deepcopy(Q_eval(Q, s))
		# the third element in each 4-list is the reward
		r = experience[i][2]
		# the 4th element is the state s'
		s_prime = experience[i][3]
		# case when s' is not terminal (!!! WARNING !!! I assume that this happens iff r == 0)
		if r == 0:
			Q_val = Q_eval(Q, s_prime)
			target = r + gamma * np.max(Q_val)
		# case when s' is terminal
		else:
			target = r
		y[i, :] = copy.deepcopy(y_pred[i,:])
		y[i, a] = copy.deepcopy(target)
	# train
	Q.fit(train_states, y, epochs = 1)

		
   
