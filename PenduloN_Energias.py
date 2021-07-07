import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

n = 3
orden = 5

ncols = 3000

Respuesta1 = np.loadtxt(f'{n}-P_1_RK{orden}.txt', usecols = [i for i in range(ncols)])
Respuesta2 = np.loadtxt(f'{n}-P_1_RK{orden + 1}.txt', usecols = [i for i in range(ncols)])
Respuesta3 = np.loadtxt(f'{n}-P_1_RK{orden + 4}.txt', usecols = [i for i in range(ncols)])


l = np.ones(n)
m = np.ones(n)

Theta1 = [rf'$\theta_{i+1}$' for i in range(n)]
Theta2 = [rf'$\theta_{i+1}$' for i in range(n)]

L = ['k--', 'k-.', 'k:', 'k-', 'k--', 'k-.', 'k:', 'k-']

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

plt.figure(figsize=(12, 8))
plt.semilogy(Respuesta1[0], np.add(T(Respuesta1), U(Respuesta1)), 'b-')
plt.semilogy(Respuesta2[0], np.add(T(Respuesta2), U(Respuesta2)), 'r-')
plt.semilogy(Respuesta3[0], np.add(T(Respuesta3), U(Respuesta3)), 'g-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$E_M$ (J)', fontsize = 18)
plt.legend(['RK4', 'RK5', 'RK8'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

#plt.figure(figsize=(12, 8))
#for i in range(n):
#    plt.semilogy(Respuesta2[0], abs(np.subtract(Respuesta1[i + 1], Respuesta2[i + 1])))
#plt.xlabel(r'$t$ (s)', fontsize = 15)
#plt.ylabel(r'$\Delta \theta$', fontsize = 15)
#plt.tick_params(labelsize = 12.5)
#plt.grid()


plt.show()