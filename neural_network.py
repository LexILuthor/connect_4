import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, losses
import tensorflow.keras.backend as kb
import copy

import play_move_functions as play
import secondary_Functions as secFun


#----------------------------------------------------------------------------------
#	CREATE CNN AND COMPILE
def create_NN(n_rows, n_columns, learn_rate = 0.01):
    Q = tf.keras.Sequential([
        layers.Conv2D(40, (2, 2), activation='sigmoid', padding = 'same', input_shape=(n_rows, n_columns, 1)),
        #layers.MaxPooling2D((2,2)),
        #layers.Dropout(0.1),
        layers.Conv2D(40, (2, 2), activation='sigmoid', padding = 'same'),
        #layers.MaxPooling2D((1,1)),
        #layers.Dropout(0.1),  # This is for regularization
        layers.Flatten(),
        #layers.Dense(200, activation='relu'),  # This can be changed later
        #layers.Dropout(0.1),
        #layers.Dense(256, activation='sigmoid'),  # This can be changed later
        #layers.Dropout(0.2),
        #layers.Dense(100, activation='sigmoid'),  # This can be changed later
        layers.Dropout(0.1),
        layers.Dense(n_columns, activation='sigmoid')  # The number of actions is equal to the number of columns
    ])
    optim = tf.keras.optimizers.SGD(
    learning_rate=learn_rate, momentum=0.0, nesterov=False, name='SGD'
	)
    Q.compile(optimizer='SGD',
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])
    return Q


#----------------------------------------------------------------------------------
#	PREDICTION
# 	Q_eval is a function that given our neural network Q 
#	and the current state s returns the values of all actions
def Q_eval(Q, current_state):
    current_state = np.array(current_state)
    (n_rows, n_columns) = np.shape(current_state)
    current_state = current_state.reshape(1, n_rows, n_columns, 1)
    return np.array(Q(current_state)[0])


#----------------------------------------------------------------------------------
#	CREATE TARGET
#	Function that creates target from experience which will be used for training
#   This is the target for Deep Q Learning
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


#----------------------------------------------------------------------------------
#	EXTRACT STATES TO TRAIN ON
def create_x_train(experience):
    len_experience = len(experience)
    n_rows = np.shape(experience[0][0])[0]
    n_columns = np.shape(experience[0][0])[1]
    S = []
    for exp in range(len_experience):
        S.append(experience[exp][0])
    S = np.array(S)
    S = S.reshape(len_experience, n_rows, n_columns, 1)
    return S


#----------------------------------------------------------------------------------
#	TRAIN
# 	The following function takes as imput the NN to train and another NN that is 
#	used to create the target (typically a "freezed" copy of the NN under train),
#	a batch of experience from memory D,
#	the discount rate gamma and the number of epochs and
# 	computes the target value, and then trains the neural network 
# 							!!! WARNING !!!
# 	I expect experience to be a list with N elements and each element is a 4-list
def train_my_NN(Q_train, Q_target, experience, gamma, n_epochs = 1):
	len_experience = len(experience)
	target = create_target(Q_target, experience, gamma)
	x_train = create_x_train(experience)
	history = Q_train.fit(x_train, target, batch_size=len_experience, epochs = n_epochs)
	#print(history.history)
	return
		



#----------------------------------------------------------------------------------
#	SAVING...
def save_NN(model, name_of_the_model="name_not_specified"):
    # save a model
    model.save('NN_parameters/Q_model_' + name_of_the_model)


#   ... and LOADING
def load_NN(name, n_rows, n_columns):
    # load a model
    # model = initialize_NN(n_rows, n_columns)
    model = tf.keras.models.load_model('NN_parameters/Q_model_' + name)
    return model
