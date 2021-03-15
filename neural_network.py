import mnist
import numpy as np
from tensorflow import keras


# Flatten the matrix.
my_train_matrix = train_matrix.flatten()

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])