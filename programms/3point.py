import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from useful_ising_functions import *

Events = 8_000
L = 10
os.chdir('..')
dir = os.getcwd()
big_folder = "RBM_Data"
small_folder = f"RBM_Data_L_{L}_Events_{Events}"
#changes directory to folder containing L= {L} and Events = {Events} data.
RBM_folder(big_folder, small_folder)
Tlist = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])
Hlist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

ksi_array = np.zeros((len(Tlist), len(Hlist)))
start = time.perf_counter()
t0 = time.perf_counter()

NT = 0
for T in Tlist:
    NH = 0
    for H in Hlist:

        G = 0
        file = f"T_{T}_H_{H}.txt"
        total_state = load_RBM_State_data(file)
        for S_flat in total_state:

            S = S_flat.reshape(L, L)

            G += corr(S, S)

        G /= Events

        G_trunc = np.zeros(int(L/2))
        points = np.linspace(1.0, int(L/2), int(L/2))

        for nn in range(int(L/2)):

            G_trunc[nn] = G[nn + 1]

        ksi_array[NT][NH] = (0.25*(np.sum((points**2)*G_trunc)))/(np.sum(G_trunc))

        if NH == 0 and NT == 0:
            t1 = time.perf_counter()
            print(f"Will take {(((t1 - t0)/60)*(len(Tlist)*len(Hlist))):.2f} min(s)")

        NH += 1
    NT += 1

os.chdir(dir)
save_RBM_State_data(f"ksi_array_L_{L}_Events_{Events}", ksi_array)

finish = time.perf_counter()
print(f"Took {(finish - start)/60:.2f} min(s)")
