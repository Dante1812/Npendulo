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
    
ticks = np.linspace(-np.pi, np.pi, 5)

"""for i in range(n):
    plt.figure(figsize = (12, 8))
    plt.plot(Acotacion(Respuesta1[i + 1]), Respuesta1[n + i + 1], 'k.', ms = 1.5)
    plt.ylabel(rf'$\omega_{i + 1}(t)$', fontsize = 18)
    plt.xlabel(rf'$\theta_{i + 1}(t)$', fontsize = 18)
    plt.tick_params(labelsize = 15)
    plt.xticks(ticks, [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    #plt.title(rf'Espacio de fases de $\theta_{i + 1}$ (1er pendulo)', fontsize = 15)
    plt.grid()

for i in range(n):
    plt.figure(figsize = (12, 8))
    plt.plot(Acotacion(Respuesta2[i + 1]), Respuesta2[n + i + 1], 'k.', ms = 1.5)
    plt.ylabel(rf'$\omega_{i + 1}(t)$', fontsize = 18)
    plt.xlabel(rf'$\theta_{i + 1}(t)$', fontsize = 18)
    plt.tick_params(labelsize = 15)
    plt.xticks(ticks, [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    #plt.title(rf'Espacio de fases de $\theta_{i + 1}$, (2do pendulo)', fontsize = 15)
    plt.grid()
"""
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
plt.ticklabel_format(useMathText=True)
#plt.title(r'Evolución de la energia total con el tiempo (2do pendulo)', fontsize = 15)
plt.tick_params(labelsize = 15)
plt.grid()

## Calculo de la distancia x e y de las masas para la animacion
def DistanciaX(n, R): 
    if n == 0:
        return np.zeros(len(R[0]))
    else:
        xn = 0
        for i in range(n):
            xn += l[i]*np.sin(R[i + 1])
        return xn

def DistanciaY(n, R):
    if n == 0:
        return np.zeros(len(R[0]))
    else:
        yn = 0
        for i in range(n):
            yn -= l[i]*np.cos(R[i + 1])
        return yn

na = 1
R1 = [[],[]]
R2 = [[],[]]

## Animacion
fig1 = plt.figure(figsize = (8, 8))
ax = fig1.gca()
def Actualizar1(i):
    R1[0].append(DistanciaX(n, Respuesta1)[na*i])
    R1[1].append(DistanciaY(n, Respuesta1)[na*i])
    R2[0].append(DistanciaX(n, Respuesta2)[na*i])
    R2[1].append(DistanciaY(n, Respuesta2)[na*i])
    if len(R1[0]) == 40:
        R1[0].pop(0)
        R1[1].pop(0)
        R2[0].pop(0)
        R2[1].pop(0)
    ax.clear()
    plt.plot(R1[0], R1[1], '-r', alpha = 0.3)
    plt.plot(R2[0], R2[1], '-b', alpha = 0.3)
    for j in range(n):
        plt.plot([DistanciaX(j, Respuesta1)[na*i], DistanciaX(j + 1, Respuesta1)[na*i]], [DistanciaY(j, Respuesta1)[na*i], DistanciaY(j + 1, Respuesta1)[na*i]], '-k', lw = 0.5)
        plt.plot(DistanciaX(j + 1, Respuesta1)[na*i], DistanciaY(j + 1, Respuesta1)[na*i], 'or', ms = 10)
        plt.plot([DistanciaX(j, Respuesta2)[na*i], DistanciaX(j + 1, Respuesta2)[na*i]], [DistanciaY(j, Respuesta2)[na*i], DistanciaY(j + 1, Respuesta2)[na*i]], '-k', lw = 0.5)
        plt.plot(DistanciaX(j + 1, Respuesta2)[na*i], DistanciaY(j + 1, Respuesta2)[na*i], 'ob', ms = 10)
    plt.title(r'$t =$'+str(round(Respuesta1[0][na*i], 3))+'s')
    plt.xlim(-(n*np.mean(l) + 0.5), (n*np.mean(l) + 0.5))
    plt.ylim(-(n*np.mean(l) + 0.5), (n*np.mean(l) + 0.5))
    plt.grid()


ani = animation.FuncAnimation(fig1, Actualizar1, range(len(Respuesta1[0])), interval = 10)

## Guardar animacion
#ani.save(f'{n}pendulo_RK{orden}.gif')
plt.show()