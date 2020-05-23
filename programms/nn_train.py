import numpy as np
import os
from sys import exit
from tqdm import tqdm
from useful_ising_functions import *

#Tlist = [np.array([1.0]), np.array([1.2]), np.array([1.4]), np.array([1.6]), np.array([1.8]), np.array([2.0]), np.array([2.2]), np.array([2.4]), np.array([2.6]), np.array([2.8]), np.array([3.0]), np.array([3.2]), np.array([3.4]), np.array([3.6]), np.array([3.8]), np.array([4.0]), np.array([4.2]), np.array([4.4]), np.array([4.6]), np.array([4.8])]
#Hlist = [np.array([0.0]), np.array([0.2]), np.array([0.4]), np.array([0.6]), np.array([0.8]), np.array([1.0]), np.array([1.2]), np.array([1.4]), np.array([1.6]), np.array([1.8])]
#Tlist = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8])
Hlist = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
Tlist = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8])
#Tlist = np.array([1.0])
#Hlist = np.array([0.0])

L = 10
Events = 2000
Epochs = 10000
batch_size = 100

os.chdir('..')
dir = os.getcwd()
#-------------------------------------------------------------------------------
#------------making of big state from RBM_Data folder---------------------------
#-------------------------------------------------------------------------------
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

batches, ids = batch_maker(train, batch_size)

one_hot_list = [one_hot_representation_from_id(Tlist, Hlist, Events, id) for id in ids]

print("Data loaded... and partitioned!")

start = time.perf_counter()
guru = my_simple_nn(100, 150, 100, 0.1, 0.0)
#guru = group_thermometer(100, 50, 10, 0.1, 0.1, mode = "sigmoid")

os.chdir(dir)

answer = yes_or_no("Do you want to use saved neural network constants?")
if answer:
    big_folder = "RBM_nn_convolutional_storage"
    small_folder = "RBM_State_finder_nn_Nv_100_Nh_150_No_100_Tstep_0.2_0.0_4.8_Hstep_0.2_0.0_1.8_Epochs_10000_batch_100"
    RBM_folder(big_folder, small_folder)
    file1 = "W_output"
    file2 = "W_hidden"
    file3 = "bias_hidden"
    file4 = "bias_output"
    guru.W_output = load_RBM_State_data(file1)
    guru.W_hidden = load_RBM_State_data(file2)
    guru.bias_hidden = load_RBM_State_data(file3)
    guru.bias_output = load_RBM_State_data(file4)
    print("Used data from"+small_folder)
else: print("Data not used!")

#-------------------------------------------------------------------------------
#-------------------------Training_phase_begins---------------------------------
#-------------------------------------------------------------------------------

print("beginning training...")

error = 0
error_keeper = 0
big_batch, big_id = batch_maker(train, int(Events/2), random = False)
print(len(big_id))

theta = [0, 0, 0, 0]

ch_time = 200.0

for epoch in tqdm(range(Epochs)):

    #starting the values for error rate calculation
    if epoch == 0:

        error_keeper = error

        sum = 0

        for b, i in zip(big_batch, big_id):
            sum += KL(one_hot_representation_from_id(Tlist, Hlist, Events, i), guru.predict(b))
        sum /= 100
        error = sum
        print(f"Starting error: {error}")
    #training session of one epoch

    for batch, one_hot in zip(batches, one_hot_list):


        theta = guru.train(batch, one_hot, *theta)


    guru.learning_momentum = guru.learning_momentum*(1.0/(1.0 + (epoch/ch_time)))


    if divmod(((epoch*100)/Epochs), 10)[1] == 0 :
        #print(f"{((epoch*100)/Epochs):.2f}% done...")

        #error rate calculation

        error_keeper = error

        sum = 0

        for b, i in zip(big_batch, big_id):
            sum += KL(one_hot_representation_from_id(Tlist, Hlist, Events, i), guru.predict(b))
        sum /= 100

        error = sum

        print(f"Error gain: {error_keeper - error}")

print(f"Final error: {error}")

finish = time.perf_counter()


print(f"Took {divmod(finish - start, 60)[0]} min(s) and {divmod(finish - start, 60)[1]} seconds")

#-------------------------------------------------------------------------------
#----------------------------Saving_nn_constants--------------------------------
#-------------------------------------------------------------------------------

os.chdir(dir)
answer = yes_or_no("Do you want to save neural network constants?")
if answer:
    big_folder = "RBM_nn_convolutional_storage"
    #small_folder = f"2.0_group_thermometer_RBM_State_finder_nn_Nv_100_Nh_50_No_10_Tstep_0.2_1.0_4.8_Epochs_10000_batch_{batch_size}_sigmoid"
    small_folder = f"RBM_State_finder_nn_Nv_100_Nh_50_No_100_Tstep_0.2_0.0_4.8_Hstep_0.2_0.0_1.8_Epochs_{20000}_batch_{batch_size}"
    RBM_folder(big_folder, small_folder)
    file1 = "W_output"
    file2 = "W_hidden"
    file3 = "bias_hidden"
    file4 = "bias_output"
    save_RBM_State_data(file1, guru.W_output)
    save_RBM_State_data(file2, guru.W_hidden)
    save_RBM_State_data(file3, guru.bias_hidden)
    save_RBM_State_data(file4, guru.bias_output)
else: print("Data Not saved...")
