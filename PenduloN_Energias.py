import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

n = 1
orden = 4

ncols = 3000

Respuesta1 = np.loadtxt(f'{n}-P_1_RK{orden}.txt', usecols = [i for i in range(ncols)])
Respuesta2 = np.loadtxt(f'{n}-P_1_RK{orden + 1}.txt', usecols = [i for i in range(ncols)])
#Respuesta3 = np.loadtxt(f'{n}-P_1_RK{orden + 4}.txt', usecols = [i for i in range(ncols)])

l = np.ones(n)
m = np.ones(n)

g = 9.81

def T(R):
    Ek = []
    for a in range(len(R[0])):
        y2 = 0
        for i in range(n):
            y1 = 0.5*m[i]
            for j in range(i + 1):
                for k in range(i + 1):
                    y2 += (l[j]*l[k]*R[j + n + 1][a]*R[k + n + 1][a]*np.cos(R[j + 1][a] - R[k + 1][a]))
            y = y1*y2
        Ek.append(y)
    return np.array(Ek)

def U(R):
    Ep = []
    for a in range(len(R[0])):
        y2 = 0
        for i in range(n):
            y1 = m[i]
            for j in range(i + 1):
                y2 += l[j]*np.cos(R[j + 1][a])
            y = -g*y1*y2
        Ep.append(y)  
    return np.array(Ep)

Et_RK4 = np.add(T(Respuesta1), U(Respuesta1))
Et_RK5 = np.add(T(Respuesta2), U(Respuesta2))
#Et_RK8 = np.add(T(Respuesta3), U(Respuesta3))

E_RK4 = abs(Et_RK4 - Et_RK4[0])
E_RK5 = abs(Et_RK5 - Et_RK5[0])
#E_RK8 = abs(Et_RK8 - Et_RK8[0])

plt.figure(figsize=(12, 8))
plt.plot(Respuesta1[0], E_RK4, 'b-')
plt.plot(Respuesta1[0], E_RK5, 'r-')
#plt.plot(Respuesta1[0], E_RK8, 'g-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$\Delta E_M$ (J)', fontsize = 18)
plt.legend(['RK4', 'RK5', 'RK8'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize=(12, 8))
plt.plot(Respuesta1[0], Et_RK4, 'b-')
plt.plot(Respuesta2[0], Et_RK5, 'r-')
#plt.plot(Respuesta3[0], Et_RK8, 'g-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$E_M$ (J)', fontsize = 18)
plt.legend(['RK4', 'RK5', 'RK8'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.show()