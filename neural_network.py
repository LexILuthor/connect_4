# !!!! WARNING !!!
# Here I just copy-pasted a CNN written for MNIST as personal exercise
# I am currently in the process of making it useful for connect 4
# !!! THIS IS WORK IN PROGRESS !!!
import numpy as np
<<<<<<< HEAD
import tensorflow as tf 
from tensorflow.keras import layers,  losses

# !!! WARNING !!! The number of actions need to be loaded and this will be loaded
num_actions = 10
=======
import tensorflow as tf
from tensorflow.keras import layers, models, losses
>>>>>>> 73046dc6386815252c4f0c784f2c47a6e863b3ff


###########################
# To Do List

# a function that returns our initialized neural network
def create_the_NN():
    pass


# a function that given training set (we still have to discuss on the type of the training set in input), computes
# the target value, and then trains the neural network using the training set
def train_my_NN():
    pass


# a function that given our neural network and the interstate/(state,action) returns the Q function
# give a better name to this function
def Q_of():
    pass


# Download the data MNIST (hand written digits)
mnist = tf.keras.datasets.mnist  # --------------------- !!!! WARNING !!! We will be using experience data from the memory D
(x_train, y_train), (
    x_test, y_test) = mnist.load_data()  # ----------------------- !!! WARNING !!! This will change as well
(rows, columns) = np.shape(x_train[0])

# We need to transform the data as a 3-tensor
x_train = x_train.reshape(-1, rows, columns, 1)
x_test = x_test.reshape(-1, rows, columns, 1)

# Create the CNN with keras
model = tf.keras.Sequential([
  layers.Conv2D( 28, (3, 3), activation='relu', input_shape= (rows,columns,1)),  
  #layers.MaxPooling2D((2,2)),            #  NOTE: Probably we will not need to do any pooling for connect 4 
  # the following layer serves to prevent overfitting by dropping out neurons randomly during training
  layers.Dropout(0.2), 				
  # We need to roll the last layer into a vector
  layers.Flatten(),
  layers.Dense(50, activation='relu'),  
  # the following layer serves to prevent overfitting by dropping out neurons randomly during training
  layers.Dropout(0.2),  
  # the last output layer is also dense 
  layers.Dense(num_actions)  			# -------------------------- !!! WARNING !!!  Instead of 10 we will use the number of actions
])

# !!! WARNING !!!     WE HAVE TO CHANGE LOSS FUNCTION
# Let's define the loss function
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

# Now we set the training configuration

#model.compile(optimizer='adam', 			# adam is a type of stochastic gradient descent 
#              loss=loss_fn,					# the loss is our loss function 
#              metrics=['accuracy'])

# Let us train the model
#model.fit(x_train, y_train, epochs=1)		# epochs are the numnbers of trainings (each train goes through the entire trainins set)
#model.evaluate(x_test,  y_test, verbose=2)	# let's evaluate the performance


predictions = model.predict(x_train)
print(predictions[0])
