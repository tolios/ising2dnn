import numpy as np
import os
from sys import exit
from useful_ising_functions import *
from tqdm import tqdm

L = 10
Events = 2000
Tlist = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])
Hlist = np.array([0.0])
Nh_collection = [9, 16, 25, 36, 49, 64, 81]
collection_error_matrix = np.zeros((7, 26, 11))
collection_cov_matrix = np.zeros((7, 26, 11))
Epochs = 8000
epoch_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*Epochs
colormap = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.9), (0.0, 0.0, 0.8), (0.0, 0.0, 0.7), (0.0, 0.0, 0.6), (0.0, 0.6, 0.6), (0.0, 0.6, 0.5), (0.0, 0.6, 0.4), (0.0, 0.6, 0.3), (0.0, 0.6,0.2), (0.0, 0.6, 0.1), (0.0,0.7, 0.0), (0.0,0.8, 0.0), (0.0, 0.9, 0.0)]
T_repr_list = [0, 2, 4, 6, 7, 8, 12, 15, 20, 25]
dm = np.zeros(26)
mm = np.zeros(26)
batch_size = Events/2

Nh = 9

#Used data from stored nn values from RBM_nn_convolutional_storage
big_folder = "RBM_nn_RBM_storage"
if Nh == 9:
    small_folder = "RBM_nn_Nv_100_Nh_9_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_8000_batch_10_learning_rate_0.001_32.62_18.68"
elif Nh == 16:
    small_folder = "RBM_nn_Nv_100_Nh_16_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_8000_batch_10_learning_rate_0.001_29.94_17.20"
elif Nh == 25:
    small_folder = "RBM_nn_Nv_100_Nh_25_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_8000_batch_10_learning_rate_0.001_26.97_15.32"
elif Nh == 36:
    small_folder = "RBM_nn_Nv_100_Nh_36_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_8000_batch_10_learning_rate_0.001_23.83_13.76"
elif Nh == 49:
    small_folder = "RBM_nn_Nv_100_Nh_49_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_8000_batch_10_learning_rate_0.001_17.73_10.37"
elif Nh == 64:
    small_folder = "RBM_nn_Nv_100_Nh_64_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_8000_batch_10_learning_rate_0.001_14.74_8.77"
elif Nh == 81:
    small_folder = "RBM_nn_Nv_100_Nh_81_Tstep_0.2_1.0_6.0_Hstep_0.0_0.0_0.0_Epochs_8000_batch_10_learning_rate_0.001_15.65_9.10"
else:
    print("No such Nh, terminating program ...")
    exit()
file1 = "W"
file2 = "bias_visual"
file3 = "bias_hidden"


os.chdir('..')
dir = os.getcwd()


#needs Nh = ... both at nn and RBM folder
genius = my_RBM(100, Nh, 0.001)

RBM_folder(big_folder, small_folder)
genius.W = load_RBM_State_data(file1)
genius.bias_visible = load_RBM_State_data(file2)
genius.bias_hidden = load_RBM_State_data(file3)


os.chdir(dir)

iterations = 1000

#making of big state from RBM_Data
collection = []
print("Loading Data...")
big_folder = "RBM_Data"
small_folder = f"RBM_Data_L_{L}_Events_{Events}"
RBM_folder(big_folder, small_folder)
sum = 0
N = 0
for T in tqdm(Tlist):
    for H in Hlist:
        file = f"T_{T}_H_{H}.txt"
        total = load_RBM_State_data(file)
        test, train = test_train_divide(total)
        m_before = np.sum(np.abs(np.sum(test, axis = 1)/(L*L)))/(Events/2)
        flowed = genius.flow(test, iterations)
        m_after = np.sum(np.abs(np.sum(flowed, axis = 1)/(L*L)))/(Events/2)
        if T == 2.4:
            m_keep = m_after
        dm[N] = np.abs(m_after - m_before)
        mm[N] = np.abs(m_after)
        N += 1

mm -= m_keep

plt.scatter(Tlist, dm)
plt.title(f"Absolute magnetization difference, iterations = {iterations} for Nh = {Nh} and Epochs = {Epochs}")
#plt.title(f"Absolute magnetization difference, iterations = {iterations} for Nh = {Nh} and untrained")
plt.xlabel("T")
plt.ylabel("|Δm|")
plt.show()

plt.scatter(Tlist, mm)
plt.title(f"m(T)-m(2.4), iterations = {iterations} for Nh = {Nh} and Epochs = {Epochs}")
#plt.title(f"m(T) reconstructed, iterations = {iterations} for Nh = {Nh} and untrained")
plt.xlabel("T")
plt.ylabel("Δm")
plt.show()


'''
os.chdir('..')
dir = os.getcwd()

os.chdir(dir+'/RBM_test_matrices')

print("Loading Data...")
N = 0
for Nh in Nh_collection:

    collection_cov_matrix[N] = load_RBM_State_data(f'Nh_{Nh}_cov_error_all_T_Epochs_{Epochs}').transpose()
    collection_error_matrix[N] = load_RBM_State_data(f'Nh_{Nh}_mean_error_all_T_Epochs_{Epochs}').transpose()
    N += 1

print("Data Loaded...")

for NNh in range(len(Nh_collection)):
    for NT in T_repr_list:

        plt.plot(epoch_points, collection_error_matrix[NNh][NT], label = f'{Tlist[NT]}')
        plt.legend(loc = "upper right")
    plt.title(f"Nh = {Nh_collection[NNh]}, error evolution during training")
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()
'''
