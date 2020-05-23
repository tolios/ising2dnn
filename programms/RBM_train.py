import numpy as np
import os
from sys import exit
from tqdm import tqdm
from useful_ising_functions import *

def miss_function(x, A, B, rho):
    return A + (B*np.exp(-(x/rho)))

Tlist = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])
#Hlist = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
Hlist = np.array([0.0])

L = 10
Events = 2000
Epochs = 8000
batch_size = 10
epoch_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*Epochs
mean_collection = np.zeros(11)
cov_collection = np.zeros(11)
epoch_points_continuous = np.linspace(epoch_points[0], epoch_points[10], 100001)

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
Nh = 16

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

        #error rate calculation

        error_keeper = error_mean

        error_mean, error_cov = genius.loss(train)

        print(N)
        mean_collection[N] = error_mean
        cov_collection[N] = error_cov
        N += 1


        if epoch != 0:
            print(f"Error gain: {error_keeper - error_mean}")
        else:

            print(f"Starting error: {error_mean:.2f} ± {error_cov:.2f}")
    #training session of one epoch
    #suffling every time
    batches, ids = batch_maker(train, batch_size, random_seed = True)

    for batch in batches:

        genius.train(batch)

error_mean, error_cov = genius.loss(train)
mean_collection[N] = error_mean
cov_collection[N] = error_cov

print(f"Final error: {error_mean:.2f} ± {error_cov:.2f}")

finish = time.perf_counter()


print(f"Took {divmod(finish - start, 60)[0]} min(s) and {divmod(finish - start, 60)[1]} seconds")

'''
popt, pcov = curve_fit(miss_function, epoch_points, mean_collection)
plt.errorbar(epoch_points, mean_collection, yerr = cov_collection, label = f"R = {popt[2]} ± {np.sqrt(pcov[2][2])}")
plt.plot(epoch_points_continuous, miss_function(epoch_points_continuous, *popt))
plt.legend(loc = 'upper right')
'''
#plt.errorbar(epoch_points, mean_collection, yerr = cov_collection)
plt.scatter(epoch_points, mean_collection)
plt.title(f"Error evolution for Nh = {Nh}, l_rate = {learning_rate}, batch_size = {batch_size}")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()



#-------------------------------------------------------------------------------
#----------------------------Saving_nn_constants--------------------------------
#-------------------------------------------------------------------------------

os.chdir(dir)
answer = yes_or_no("Do you want to save the RBM neural network constants?")
if answer:
    big_folder = "RBM_nn_RBM_storage"
    #small_folder = f"2.0_group_thermometer_RBM_State_finder_nn_Nv_100_Nh_50_No_10_Tstep_0.2_1.0_4.8_Epochs_10000_batch_{batch_size}_sigmoid"
    small_folder = f"RBM_nn_Nv_100_Nh_{Nh}_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_{Epochs}_batch_{batch_size}_learning_rate_{learning_rate}_{error_mean:.2f}_{error_cov:.2f}"
    #small_folder = "test"
    RBM_folder(big_folder, small_folder)
    file1 = "W"
    file2 = "bias_visual"
    file3 = "bias_hidden"
    save_RBM_State_data(file1, genius.W)
    save_RBM_State_data(file2, genius.bias_visible)
    save_RBM_State_data(file3, genius.bias_hidden)
else: print("Data Not saved...")
