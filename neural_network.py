# !!!! WARNING !!!
# Here I just copy-pasted a CNN written for MNIST as personal exercise
# I am currently in the process of making it useful for connect 4
# !!! THIS IS WORK IN PROGRESS !!!
import numpy as np
<<<<<<< HEAD
import tensorflow as tf 
from tensorflow.keras import layers,  losses
import tensorflow.keras.backend as kb
=======
import tensorflow as tf
from tensorflow.keras import layers, losses

>>>>>>> ba166970f7d687a89ec9442e7bf8112d2e17cb9e

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

    return Q


# Q_eval is a function that given our neural network Q and the current state s returns the values of all actions
def Q_eval(Q, current_state):
    # we need to reshape the state into a tensor
    current_state = np.array(current_state)
    (n_rows, n_columns) = np.shape(current_state)
    current_state = current_state.reshape(1, n_rows, n_columns, 1)
    return Q.predict(current_state)


###########################
# To Do List


<<<<<<< HEAD
# a function that given a batch of training set coming from memory D, and the discount rate gamma
# computes the target value, and then trains the neural network 
# !!! WARNING !!!
# I expect experience_batch to be a list with N elements and each element is a 4-list
def train_my_NN(experience_batch, Q, gamma):
	# lenght of experience batch
	len_batch = len(experience_batch)

	# initialize target
	y = np.zeros(len_batch)
	# initialize prediction
	y_pred = np.zeros(len_batch)
	for i in range(len_batch):
		# the first element in each 4-list is the state s
		s = experience_batch[i][0]
		# the second element is the action a
		a = experience_batch[i][1]
		# let's compute the predicted value of (s,a)
		y_pred[i] = Q_eval(Q, s)[a]

		# the third element in each 4-list is the reward
		r = experience_batch[i][2]
		# the 4th element is the state s'
		s_prime = experience_batch[i][3]
		# case when s' is not terminal (!!! WARNING !!! I assume that this happens iff r == 0)
		if r == 0:
			Q_val = Q_eval(Q, s_prime)
			y[i] = r + gamma * np.max(Q_val)
		# case when s' is terminal
		else:
			y[i] = r

	# define the loss function
	error = (y - y_pred)
	loss = np.inner(error,error)

	# Now we set the training configuration
	Q.compile(optimizer='adam', 			# adam is a type of stochastic gradient descent 
              loss=loss,					# the loss is our loss function 
              metrics=['accuracy'])

	# train
	Q

		
    

=======
# a function that given training set (we still have to discuss on the type of the training set in input), computes
# the target value, and then trains the neural network using the training set
def train_my_NN():
    pass
>>>>>>> ba166970f7d687a89ec9442e7bf8112d2e17cb9e
