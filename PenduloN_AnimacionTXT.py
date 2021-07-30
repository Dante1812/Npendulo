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