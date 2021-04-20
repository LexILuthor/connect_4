import os
import numpy as np
import copy
import random
import tensorflow as tf
import time
#---------------------------------
import secondary_Functions as secFun
import testing_functions as test
import neural_network as nn
import memory 
import training_functions as train

##########################################################################################

# Available NN: 
# test  (dense)     exagerated nunber of neurons and layers
# shallow_1         shallow (dense) nn: in+out             rate of success vs random: ~90%
# shallow_2         as shallow_1
# 
# CNN_1             cnn: in+con+pool+dense(large)+out
# CNN_2             cnn: in+con+pool+con+pool+dense(small)+out
# CNN_3             cnn: in + con + dense + dense + out
# CNN_4             cnn: as CNN_3

# dense_2_hidden_pl1    dense with 2 hidden layers (only 100 neur per layer)
# dense_2_hidden_pl2    dense wiht 2 hidden layers (only 100 neur per layer)
#
# dense_2_hidden_pl1    dense with 2 hidden layers (256 neur per layer)
# dense_2_hidden_pl2    dense with 2 hidden layers (256 neur per layer)
#
# freeze            dense with 1 hidden layer (trained only with freeze method) vs random ~77%

# dense_sigmoid_pl1    dense with 2 hidden layers (256 neur per layer)
# dense_sigmoid_pl2    dense with 2 hidden layers (256 neur per layer)
#
# cnn_sigmoid_pl1    CNN + 2 dense hidden layers (256 neur per layer)
# cnn_sigmoid_pl2    CNN + 2 dense hidden layers (256 neur per layer)

# cnn_pl1    CNN + CNN + outp
# cnn_pl2    CNN + CNN + outp


def main():
    N = 10
    memory_size = 100
    n_rows = 6
    n_columns = 7
    Q_player1 = nn.load_NN("cnn_pl1", n_rows, n_columns)
    Q_player2 = nn.load_NN("cnn_pl2", n_rows, n_columns)
    #Q_player1 = nn.create_NN(n_rows, n_columns)
    #Q_player2 = nn.create_NN(n_rows, n_columns)
    # compile
    Q_player1.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])
    Q_player2.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])
    
    

#-----------------------------------------------------------------------------------
    precision_VS_random = 0
    while precision_VS_random < 100:
        history_player1 = []
        precision_player1 = 0
        # initialize memory for player 1 
        mem1 = memory.create_memory_player1(
        memory_size, 
        n_rows, 
        n_columns, 
        Q_agent = Q_player1, 
        Q_environment = Q_player2,
        epsilon = 0.2)
        # debugging
        os.system("clear")
        #print(mem1)
        #time.sleep(5)
        # Training cycle for player 1
        while precision_player1 < 99 :
            #Q_player1 = nn.load_NN("dense_leaky_relu_pl1", n_rows, n_columns)
            Q_player1 = copy.copy(train.freeze_train_player1(
            Q_player1,
            Q_player2,
            n_rows,
            n_columns, 
            memory_player1 = mem1,
            episodic = False,
            number_of_moves = 1000,
            freeze_steps_player1 = 5,
            batch_size_player1 = 50,
            rewards_Wi_Lo_Dr_De = [10, -10, 1, 0],
            epsilon_player1 = 0.1,
            gamma = 0.9,
            n_epochs = 3
            ))
            nn.save_NN(Q_player1, "cnn_pl1")
            precision_player1 = copy.copy(test.test_vs_AI_player1(
                Q_player1, 
                Q_player2, 
                n_rows, 
                n_columns,

                number_of_moves = 100
                ))
            history_player1.append(round(precision_player1, 2))
            print(history_player1)
            # keep track of precision vs random
            precision_VS_random = test.test_vs_random(Q_player1, n_rows, n_columns)
            time.sleep(2)
            #os.system('spd-say "Press Enter to continue"')
            #input("Press Enter to continue...")
#-----------------------------------------------------------------------
        history_player2 = []
        precision_player2 = 0
        # Initialize memory for player 2
        mem2 = memory.create_memory_player2(
        memory_size, 
        n_rows, 
        n_columns, 
        Q_agent = Q_player2, 
        Q_environment = Q_player1, 
        epsilon = 0.2)
        os.system("clear")
        # debugging
        # print(mem2)
        # time.sleep(60)
        # Training cycle for player 2
        while precision_player2 < 95 :
            #Q_player2 = nn.load_NN("dense_leaky_relu_pl2", n_rows, n_columns)
            Q_player2 = copy.copy(train.freeze_train_player2(
            Q_player1,
            Q_player2,
            n_rows,
            n_columns, 
            memory_player2 = mem2,
            episodic = False,
            number_of_moves = 1000,
            freeze_steps_player2 = 5,
            batch_size_player2 = 100,
            rewards_Wi_Lo_Dr_De = [10, -10, 1, 0],
            epsilon_player2 = 0.1,
            gamma = 0.9,
            n_epochs = 4
            ))
            nn.save_NN(Q_player2, "cnn_pl2")
            precision_player2 = copy.copy(test.test_vs_AI_player2(
                Q_player1, 
                Q_player2, 
                n_rows, 
                n_columns, 
                number_of_moves = 100
                ))
            history_player2.append(round(precision_player2,2))
            print(history_player2)
            test.test_vs_random(Q_player2, n_rows, n_columns)
            time.sleep(2)
            #os.system('spd-say "Press Enter to continue"')
            #tinput("Press Enter to continue...")



if __name__ == '__main__':
    main()
