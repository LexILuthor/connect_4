# !!!! WARNING !!!
# Here I just copy-pasted a CNN written for MNIST as personal exercise
# I am currently in the process of making it useful for connect 4
# !!! THIS IS WORK IN PROGRESS !!!
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses


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
    (n_rows, n_columns) = np.shape(current_state)
    current_state = current_state.reshape(1, n_rows, n_columns, 1)
    return Q.predict(current_state)


###########################
# To Do List


# a function that given training set (we still have to discuss on the type of the training set in input), computes
# the target value, and then trains the neural network using the training set
def train_my_NN():
    pass
