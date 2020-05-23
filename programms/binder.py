import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.optimize import curve_fit
from Metropolis_final import *

def linear(x,a,b):
    return a*x + b

def Temp(al, ak, bl, bk):
    return (bl - bk)/(ak - al)

def DTemp(al, ak, bl, bk, w, dak, dbl, dbk):
    m1 = ((bl/(ak - al))**2)* dbk
    m2 = ((bk/(ak - al))**2)* dbl
    m3 = (((bk-bl)/((ak - al)**2))**2)* w
    m4 = (((bk-bl)/((ak - al)**2))**2)* dak
    return np.sqrt(m1 + m2 + m3 + m4)

Measurements = 20
Events = 3000
Therm_t = 7000
Decorr_t = 10
T_range = np.linspace(2.265, 2.272, num = 8)
#sets how many binder plots
L_set = [32, 64]
color_set = ['r', 'b']
color_set_plot = ['red', 'blue']
#############################


A = np.zeros(3)
B = np.zeros(3)
DA = np.zeros(3)
DB = np.zeros(3)

H = 0
Nc = 0

for L in L_set:
    binder = np.zeros(T_range.shape[0])
    err_binder = np.zeros(T_range.shape[0])
    for mm in range(Measurements):
        if ((mm*100)/Measurements) % 5 == 0: print((mm*100)/Measurements)
        N = 0

        for T in T_range:



            S = np.ones((L, L), dtype=int)
            #thermalization
            for tht in range(Therm_t):
                S = metro_swipe(S, T, H)

            s_second = 0
            s_fourth = 0
            for ee in range(Events):

                m = magnetization(S)
                s_second += m**2
                s_fourth += m**4



                for dct in range(Decorr_t):
                    S = metro_swipe(S, T, H)

            s_second /= Events
            s_fourth /= Events


            binder[N] += 1 - ((s_fourth)/(3*(s_second**2)))
            err_binder += (1 - ((s_fourth)/(3*(s_second**2))))**2
            N += 1

    binder /= Measurements
    err_binder /= Measurements
    err_binder -= binder**2
    err_binder = np.sqrt(err_binder)

    plt.errorbar(T_range, binder, yerr = err_binder, fmt = 'o', ecolor = color_set[Nc], label = str(L))
    popt, pcov = curve_fit(linear, T_range, binder)
    plt.plot(T_range, linear(T_range, *popt), color = color_set_plot[Nc])

    A[Nc] = popt[0]
    B[Nc] = popt[1]
    DA[Nc] = pcov[0,0]
    DB[Nc] = pcov[1,1]

    Nc += 1



#calculation of Tc (r,g,b)
print("#################")
print("Tc from red and green:")
print(Temp(A[0],A[1],B[0],B[1]),"±",DTemp(A[0], A[1], B[0], B[1], DA[0], DA[1], DB[0], DB[1]))
"""""""""
print("Tc from red and blue:")
print(Temp(A[0],A[2],B[0],B[2]),"±",DTemp(A[0], A[2], B[0], B[2], DA[0], DA[2], DB[0], DB[2]))
print("#################")
print("Tc from blue and green:")
print(Temp(A[2],A[1],B[2],B[1]),"±",DTemp(A[2], A[1], B[2], B[1], DA[2], DA[1], DB[2], DB[1]))
"""""""""""


plt.legend(loc = 'upper right')
plt.title("Binder Cumulant plot")
plt.show()
