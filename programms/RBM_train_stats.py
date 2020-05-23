import numpy as np
import os
from sys import exit
from tqdm import tqdm
from useful_ising_functions import *

def miss_function(x, A, B, rho):
    return A + (B*np.exp(-(x/rho)))

#Tlist = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])
Tlist = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])
#Hlist = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
Hlist = np.array([0.0])

L = 10
Events = 2000
Epochs = 8000
batch_size = 10
epoch_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*Epochs
collection_error_matrix = np.zeros((1, 11, 16))
collection_cov_matrix = np.zeros((1, 11, 16))

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
print(train.shape)

print("Data loaded... and partitioned!")

start = time.perf_counter()
learning_rate = 0.001
Nh_collection = np.array([81])
start_error = np.zeros(7)
final_error = np.zeros(7)
M = 0
simple_batches, simple_ids = batch_maker(train, 1000)

for Nh in Nh_collection:
    print(f"Start of Nh_{Nh}")
    genius = my_RBM(100, Nh, learning_rate)

    os.chdir(dir)

    #answer = yes_or_no("Do you want to use saved neural network constants?")
    answer = False
    if answer:
        pass
    else:
        pass

    #-------------------------------------------------------------------------------
    #-------------------------Training_phase_begins---------------------------------
    #-------------------------------------------------------------------------------

    print("beginning training...")

    error_mean = 0
    error_cov = 0
    error_keeper = 0
    N = 0
    for epoch in tqdm(range(Epochs)):

        if divmod(((epoch*100)/Epochs), 10)[1] == 0 :

            error_keeper = error_mean

            NT = 0
            for T_batch in simple_batches:

                mean, cov = genius.loss(T_batch)

                collection_error_matrix[M][N][NT] = mean
                collection_cov_matrix[M][N][NT] = cov

                NT += 1

            error_mean = np.sum(collection_error_matrix[0][N])/16
            error_cov = np.sum(collection_cov_matrix[0][N])/16
            if epoch != 0:
                print(f"Error gain: {error_keeper - error_mean}")
            else:

                print(f"Starting error: {error_mean:.2f} ± {error_cov:.2f}")
            print(N)
            N += 1
        #training session of one epoch
        #suffling every time
        batches, ids = batch_maker(train, batch_size, random_seed = True)

        for batch in batches:

            genius.train(batch)

    error_mean, error_cov = genius.loss(train)
    print(error_mean)

    NT = 0
    for T_batch in simple_batches:

        mean, cov = genius.loss(T_batch)

        collection_error_matrix[M][10][NT] = mean
        collection_cov_matrix[M][10][NT] = cov

        NT += 1

    #saving data
    os.chdir(dir)
    #answer = yes_or_no("Do you want to save the RBM neural network constants?")
    answer = True
    if answer:
        big_folder = "RBM_nn_RBM_storage"
        #small_folder = f"2.0_group_thermometer_RBM_State_finder_nn_Nv_100_Nh_50_No_10_Tstep_0.2_1.0_4.8_Epochs_10000_batch_{batch_size}_sigmoid"
        small_folder = f"RBM_nn_Nv_100_Nh_{Nh}_Tstep_0.2_3.0_6.0_Hstep_0.0_0.0_0.0_Epochs_{Epochs}_batch_{batch_size}_learning_rate_{learning_rate}_{error_mean:.2f}_{error_cov:.2f}"
        #small_folder = "test"
        RBM_folder(big_folder, small_folder)
        file1 = "W"
        file2 = "bias_visual"
        file3 = "bias_hidden"
        save_RBM_State_data(file1, genius.W)
        save_RBM_State_data(file2, genius.bias_visible)
        save_RBM_State_data(file3, genius.bias_hidden)
    else: print("Data Not saved...")



    M += 1

os.chdir(dir)
os.chdir(dir+"/RBM_test_matrices")

for M in range(Nh_collection.shape[0]):
    save_RBM_State_data(f"Hot_T_Nh_{Nh_collection[M]}_mean_error_all_T_Epochs_{Epochs}", collection_error_matrix[M])
    save_RBM_State_data(f"Hot_T_Nh_{Nh_collection[M]}_cov_error_all_T_Epochs_{Epochs}", collection_cov_matrix[M])


finish = time.perf_counter()
print(f"Took {divmod(finish - start, 60)[0]} min(s) and {divmod(finish - start, 60)[1]} seconds")
