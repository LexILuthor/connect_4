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
        layers.Conv2D(10, (4, 4), strides=1, activation='relu', input_shape=(n_rows, n_columns, 1,)),
        layers.Conv2D(25, (2, 2), strides=1, padding='valid', activation='relu', input_shape=(n_rows, n_columns, 1)),
        layers.MaxPooling2D(pool_size=(1, 1)),
        layers.Flatten(),
        layers.Dense(15, activation='softmax'),
        layers.Dense(20, activation='relu'),  # This can be changed later
        layers.Dropout(0.2),  # The number of actions is equal to the number of columns
        layers.Dense(1)
    ])
    Q.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])
    return Q


# Q_eval is a function that given our neural network Q and the current state s returns the values of all actions
def Q_eval(Q, current_state):
    # we need to reshape the state into a tensor
    (n_rows, n_columns) = np.shape(current_state[0])
    state = np.reshape(current_state, (len(current_state), n_rows, n_columns, 1))
    result = Q(state)
    result = result[:, 0].numpy()
    return result


###########################
# To Do List


# a function that given training set (we still have to discuss on the type of the training set in input), computes
# the target value, and then trains the neural network using the training set
def train_my_NN(Q, SA_intermediate_state, r, S_prime, agent_color=1):
    gamma = 0.9
    y_target_state = [secFun.compute_target_y(Q, SA_intermediate_state[i], r[i], S_prime[i], agent_color, gamma) for i
                      in range(len(r))]
    y_target_state = np.array(y_target_state)
    SA_intermediate_state = np.array(SA_intermediate_state)
    n_rows, n_columns = np.shape(SA_intermediate_state[0])
    SA_intermediate_state = np.reshape(SA_intermediate_state, (SA_intermediate_state.shape[0], n_rows, n_columns, 1))

    Q.fit(x=SA_intermediate_state, y=y_target_state)


def save_NN(model, name_of_the_model="name_not_specified"):
    # save a model
    model.save('NN_parameters/Q_model_' + name_of_the_model)


def load_NN(name, n_rows, n_columns):
    # load a model
    # model = initialize_NN(n_rows, n_columns)
    model = tf.keras.models.load_model('NN_parameters/Q_model_' + name)
    return model
