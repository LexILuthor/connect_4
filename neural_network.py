# !!!! WARNING !!!
# Here I just copy-pasted a CNN written for MNIST as personal exercise
# I am currently in the process of making it useful for connect 4
# !!! THIS IS WORK IN PROGRESS !!!
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
import copy

import secondary_Functions as secFun


# a function that returns our initialized neural network
def initialize_NN(n_rows, n_columns):
    Q = tf.keras.Sequential([
        layers.Conv2D(10, (3, 3), activation='relu', input_shape=(n_rows, n_columns, 1)),
        layers.Dropout(0.1),  # This is for regularization
        layers.Flatten(),
        layers.Dense(20, activation='relu'),  # This can be changed later
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    Q.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])
    return Q


# Q_eval is a function that given our neural network Q and the current state s returns the values of all actions
def Q_eval(Q, current_state):
    # we need to reshape the state into a tensor
    (n_rows, n_columns) = np.shape(current_state)
    return Q.predict(np.reshape(current_state, (1, n_rows, n_columns, 1)))


###########################
# To Do List


# a function that given training set (we still have to discuss on the type of the training set in input), computes
# the target value, and then trains the neural network using the training set
def train_my_NN(Q, SA_intermediate_state, r, S_prime, agent_color=1):
    gamma = 1
    print("start")
    y_target_state = [secFun.compute_target_y(Q, SA_intermediate_state[i], r[i], S_prime[i], agent_color, gamma) for i
                      in range(len(r))]
    print("end")
    y_target_state = np.array(y_target_state)
    SA_intermediate_state = np.array(SA_intermediate_state)
    n_rows, n_columns = np.shape(SA_intermediate_state[0])
    SA_intermediate_state = np.reshape(SA_intermediate_state, (SA_intermediate_state.shape[0], n_rows, n_columns, 1))

    Q.fit(x=SA_intermediate_state, y=y_target_state)


def save_NN(model):
    # save a model
    model.save('NN_parameters/Q.h5')  # creates a HDF5 file 'my_model.h5'
    model.save_weights('NN_parameters/Q_weights.h5')


def load_NN(n_rows, n_columns):
    # load a model
    model = initialize_NN(n_rows, n_columns)
    model.load_weights('NN_parameters/Q_weights.h5')
    return model
