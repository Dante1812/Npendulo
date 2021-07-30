import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 1 ## Valor de N
orden = 4 ## Orden para el metodo de Runge - Kutta 

ncols = 3000 ## Colas a usar del .txt
## .txt a analizar
Respuesta1 = np.loadtxt(f'{n}-P_1_RK{orden}.txt', usecols = [i for i in range(ncols)])
Respuesta2 = np.loadtxt(f'{n}-P_2_RK{orden}.txt', usecols = [i for i in range(ncols)])

l = np.ones(n) ## Valores para las masas 
m = np.ones(n) ## Valores para las longitudes de la cuerda
g = 9.81 ## Valor de g

## Para las leyendas de las gráficas
Theta1 = [rf'$\theta_{i+1}$' for i in range(n)] 
Theta2 = [rf'$\theta_{i+1}^*$' for i in range(n)] 
L1 = ['b-', 'r-', 'g-', 'm-', 'y-', 'c-', 'k-']
L2 = ['b:', 'r:', 'g:', 'm:', 'y:', 'c:', 'k:']

def Acotacion(R):
    for i in range(len(R)):
        while R[i] > np.pi:
            R[i] = R[i] - 2*np.pi
        while R[i] < -np.pi:
            R[i] = R[i] + 2*np.pi
    return R

plt.figure(figsize = (12, 8))
for i in range(n):
    plt.plot(Respuesta1[0], Respuesta1[i+1], L1[i])
plt.legend(Theta1, fontsize = 18)
plt.xlabel(rf'$t$ (s)', fontsize = 18)
plt.ylabel(rf'$\theta(t)$ (rad)', fontsize = 18)
plt.tick_params(labelsize = 15)
#plt.title(rf'Evolucion de $\theta$ con el tiempo (1er pendulo)', fontsize = 15)
plt.grid()

plt.figure(figsize = (12, 8))
for i in range(n):
    plt.plot(Respuesta2[0], Respuesta2[i+1], L2[i])
plt.legend(Theta2, fontsize = 18)
plt.xlabel(rf'$t$ (s)', fontsize = 18)
plt.ylabel(rf'$\theta(t)$ (rad)', fontsize = 18)
plt.tick_params(labelsize = 15)
#plt.title(rf'Evolucion de $\theta$ con el tiempo (2do pendulo)', fontsize = 15)
plt.grid()

for i in range(n):
    plt.figure(figsize = (12, 8))
    plt.plot(Respuesta1[0], Respuesta1[i+1], L1[i])
    plt.plot(Respuesta2[0], Respuesta2[i+1], L2[i])
    plt.legend([rf'$\theta_{i + 1}$', rf'$\theta_{i + 1}^*$'], fontsize = 18)
    plt.xlabel(rf'$t$ (s)', fontsize = 18)
    plt.ylabel(rf'$\theta_{i + 1}(t)$ (rad)', fontsize = 18)
    plt.tick_params(labelsize = 15)
    plt.grid()
    #plt.title(rf'Comparacion de $\theta_{i + 1}$ vs tiempo para ambos pendulos', fontsize = 15)

plt.figure(figsize = (12, 8))
for j in range(n):
    plt.plot(Respuesta1[0], Respuesta1[j+1], L1[j])
    plt.plot(Respuesta2[0], Respuesta2[j+1], L2[j])
plt.legend([r'$\theta_{i}$', r'$\theta_{i}^{*}$'], fontsize = 18)
plt.xlabel(rf'$t$ (s)', fontsize = 18)
plt.ylabel(rf'$\theta(t)$ (rad)', fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()
#plt.title(rf'Comparacion de $\theta$ vs tiempo para ambos pendulos', fontsize = 15)

def T(R): ## Funcion que calcula la energia cinetica
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

def U(R): ## Funcion que calcula la energia potencial
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
plt.plot(Respuesta1[0], T(Respuesta1), 'b-')
plt.plot(Respuesta1[0], U(Respuesta1), 'r-')
plt.plot(Respuesta1[0], np.add(T(Respuesta1), U(Respuesta1)), 'g-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$E$ (J)', fontsize = 18)
#plt.title(r'Evolución de la energia con el tiempo (1er pendulo)', fontsize = 15)
plt.legend([r'$T$', r'$U$', r'$E_M$'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize=(12, 8))
plt.plot(Respuesta2[0], np.add(T(Respuesta1), U(Respuesta1)), 'b-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$E_M$ (J)', fontsize = 18)
plt.ticklabel_format(useMathText=True)
#plt.title(r'Evolución de la energia total con el tiempo (1er pendulo)', fontsize = 15)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize=(12, 8))
plt.plot(Respuesta2[0], T(Respuesta2), 'b-')
plt.plot(Respuesta2[0], U(Respuesta2), 'r-')
plt.plot(Respuesta2[0], np.add(T(Respuesta2), U(Respuesta2)), 'g-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$E$ (J)', fontsize = 18)
#plt.title(r'Evolución de la energia con el tiempo (2do pendulo)', fontsize = 15)
plt.legend([r'$T$', r'$U$', r'$E_M$'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize=(12, 8))
plt.plot(Respuesta2[0], np.add(T(Respuesta2), U(Respuesta2)), 'b-')
plt.xlabel(r'$t$ (s)', fontsize = 18)
plt.ylabel(r'$E_M$ (J)', fontsize = 18)
#plt.title(r'Evolución de la energia total con el tiempo (2do pendulo)', fontsize = 15)
plt.tick_params(labelsize = 15)
plt.grid()

plt.show()