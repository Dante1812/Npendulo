import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

n = 10
orden = 5

ncols_0 = 3000

Respuesta1_comp = np.loadtxt(f'{n}-P_1_RK{orden}.txt', usecols = [i for i in range(ncols_0)])
Respuesta2_comp = np.loadtxt(f'{n}-P_2_RK{orden}.txt', usecols = [i for i in range(ncols_0)])

ncols = 1100

Respuesta1 = np.loadtxt(f'{n}-P_1_RK{orden}.txt', usecols = [i for i in range(ncols)])
Respuesta2 = np.loadtxt(f'{n}-P_2_RK{orden}.txt', usecols = [i for i in range(ncols)])

def MinimosCuadrados(puntosx, puntosy):
    n = len(puntosx)
    fit = []
    fit_x = np.linspace(np.amin(puntosx), np.amax(puntosx), 1000) 
    a1 = (n*sum(np.multiply(puntosx, puntosy)) - sum(puntosx)*sum(puntosy))/(n*sum(np.power(puntosx, 2)) - (sum(puntosx))**2)
    a0 = np.mean(puntosy) - a1*np.mean(puntosx)
    for x in fit_x:
        y = a0 + a1*x
        fit.append(y)
    return a0, a1, fit_x, fit

T = np.zeros((n, ncols))
W = np.zeros((n, ncols))
T1 = np.zeros((n, ncols_0))
W1 = np.zeros((n, ncols_0))
y, y1 = 0, 0
for i in range(n):
    T[i] = abs(np.subtract(Respuesta1[i + 1], Respuesta2[i + 1]))
    T1[i] = abs(np.subtract(Respuesta1_comp[i + 1], Respuesta2_comp[i + 1]))
    W[i] = abs(np.subtract(Respuesta1[n +i + 1], Respuesta2[n + i + 1]))
    W1[i] = abs(np.subtract(Respuesta1_comp[n + i + 1], Respuesta2_comp[n + i + 1]))
    y += T[i]**2 + W[i]**2
    y1 += T1[i]**2 + W1[i]**2
Z = np.sqrt(y)
Z1 = np.sqrt(y1)

a0, a1, fit_x, fit = MinimosCuadrados(Respuesta1[0], np.log(Z))
t = 1/a1*np.log(1/Z1[0])

plt.figure(figsize=(12, 8))
plt.semilogy(Respuesta1_comp[0], Z1, 'b-')
plt.semilogy(fit_x, np.exp(fit), 'r-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$\Delta r$', fontsize = 18)
plt.legend(['Datos', 'Ajuste'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

print(f'Separacion inicial: {Z1[0]}')
print(f'Mayor exp de Lyapunov: {a1}')
print(f'Tiempo de prediccion: {t}')
plt.show()