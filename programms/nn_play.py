import numpy as np
import os
from sys import exit
from useful_ising_functions import *
from tqdm import tqdm
'''
L = 10
#Hlist = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
#Tlist = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6])
Tlist = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8])
Hlist = np.array([0.0])
Events = 2000
batch_size = 1

big_folder = "RBM_nn_convolutional_storage"
small_folder = "group_thermometer_RBM_State_finder_nn_Nv_100_Nh_50_No_20_Tstep_0.2_1.0_4.8_Epochs_10000_batch_100_sigmoid"
file1 = "W_output"
file2 = "W_hidden"
file3 = "bias_hidden"
file4 = "bias_output"
#guru is informed of his knowledge
guru = group_thermometer(100, 20, 20, 0.1, 0.0)
os.chdir('..')
dir = os.getcwd()
RBM_folder(big_folder, small_folder)
guru.W_output = load_RBM_State_data(file1)
guru.W_hidden = load_RBM_State_data(file2)
guru.bias_hidden = load_RBM_State_data(file3)
guru.bias_output = load_RBM_State_data(file4)

T = 1.0
H = 0.0

os.chdir(dir)

#data loaded from one T, H file
print("Loading Data...")
big_folder = "RBM_data"
small_folder = f"RBM_Data_L_{L}_Events_{Events}"
RBM_folder(big_folder, small_folder)
file = f"T_{T}_H_{H}.txt"
state = load_RBM_State_data(file)

test, train = test_train_divide(state)
print("Data loaded... and partitioned!")

desired_state = one_hot_representation(Tlist, Hlist, T, H)

prediction = guru.predict(test)

#print((prediction - desired_state).shape)
arr = prediction


plt.imshow(arr1, cmap='gray', extent=[0.0, 1.8, 4.6, 1.0])
#plt.imshow(arr2, cmap='gray', extent=[0.0, 1.8, 4.6, 1.0])
plt.colorbar()
plt.xlabel('H')
plt.ylabel('T')

#plt.scatter(Tlist, arr5/arr3)

plt.scatter(Tlist, arr)
plt.show()
'''
a = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
print(a*1000)
