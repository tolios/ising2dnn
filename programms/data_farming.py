import numpy as np
import time
import concurrent.futures as ftr
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from useful_ising_functions import *

def vnice(x,A,D):
    return (A*(x**(-2*D)))

os.chdir('..')

Events = 2000
L = 32
Decorr_t = 10
Therm_t = 8000
Tlist = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])
Hlist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])


start = time.perf_counter()
#goes to RBM_data directory
RBM_folder("RBM_Data", f"RBM_Data_L_{L}_Events_{Events}")
t0 = time.perf_counter()
for T in Tlist:
    for H in Hlist:
        RBM_make_text(L, Events, T, H, Therm_t, Decorr_t)
        if T == 1.0 and H == 0.0:
            t1 = time.perf_counter()
            print(f"The process will be finished in about {((t1-t0)/60)*((len(Tlist)*len(Hlist))-1):.2f} min(s)")
finish = time.perf_counter()
print(f"Took {(finish - start)/60:.2f} min(s)")
#CREATE MEASUREMENTS???
