import numpy as np
import os
from sys import exit
from useful_ising_functions import *

L = 10
Events = 2000
Hlist = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
Tlist = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6])
#Tlist = np.array([1.0])
#Hlist = np.array([0.0])
batch_size = Events/2

#Used data from stored nn values from RBM_nn_convolutional_storage
big_folder = "RBM_nn_convolutional_storage"
small_folder = "RBM_State_finder_nn_Nv_100_Nh_150_No_100_Tstep_0.4_1.0_4.6_Hstep_0.2_0.0_1.8_Epochs_10000_batch_100_sigmoid"
file1 = "W_output"
file2 = "W_hidden"
file3 = "bias_hidden"
file4 = "bias_output"

os.chdir('..')
dir = os.getcwd()

guru = my_simple_nn(100, 150, 100, 0.1)
RBM_folder(big_folder, small_folder)
guru.W_output = load_RBM_State_data(file1)
guru.W_hidden = load_RBM_State_data(file2)
guru.bias_hidden = load_RBM_State_data(file3)
guru.bias_output = load_RBM_State_data(file4)

os.chdir(dir)


#making of big state from RBM_Data
collection = []
print("Loading Data...")
big_folder = "RBM_Data"
small_folder = f"RBM_Data_L_{L}_Events_{Events}"
RBM_folder(big_folder, small_folder)
for T in Tlist:
    for H in Hlist:
        file = f"T_{T}_H_{H}.txt"
        hello = load_RBM_State_data(file)
        collection.append(hello)

total_state = np.concatenate(collection, axis = 0)

test, train = test_train_divide(total_state)

batches, ids = batch_maker(train, batch_size, random = False)

one_hot_list = [one_hot_representation_from_id(Tlist, Hlist, Events, id) for id in ids]

error = 0
N = 0
for batch, one_hot in zip(batches, one_hot_list):
    if N == 0: print(batch.shape)
    prediction = np.sum(guru.predict(batch), axis = 0)/(Events/2)
    prediction = max_representation(prediction)
    error += KL(np.sum(one_hot, axis = 0)/(Events/2), prediction)
    N += 1

error /= N

print(error)
print(N)




#plt.scatter(Tlist, prob)
#plt.show()
#Data loaded to nn
