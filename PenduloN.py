#!/usr/bin/python3
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

n = 6 ## Valor de N
g = 9.81 ## Valor de g
orden = 5 ## Orden para el metodo de Runge - Kutta

m = np.ones(n) ## Valores para las masas
l = np.ones(n) ## Valores para las longitudes de la cuerda

def Funcion(t, X, n, l, m): ## Funcion que calcula las 2n EDO's de primer orden
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    e = np.ones((n, 1))*-1
    for i in range(n):
        for j in range(n):
            for k in range(max(i,j),n):
                A[i][j] += m[k]
                B[i][j] += m[k]
            if i == j:
                A[i][j] *= l[j]
                B[i][j] *= g * np.sin(X[i])
            else:
                A[i][j] *= l[j]*np.cos(X[i]-X[j])
                B[i][j] *= l[j]*X[n + j]**2*np.sin(X[i]-X[j])
    M = np.dot(np.linalg.inv(A),B)
    M1 = np.dot(M, e)
    W = np.zeros(n)
    T = np.zeros(n)
    for i in range(n):
        W[i] = M1[i]
        T[i] = X[n + i]
    return np.append(T, W)

def RK4(Funcion, n, l, m, X0, t, t_max, h): ## Metodo RK4
    X = np.array(X0)
    Valores = []
    for t in np.arange(t, t_max, h):
        K1 = Funcion(t, X, n, l, m)
        K2 = Funcion(t + h/2, X + h*K1/2, n, l, m)
        K3 = Funcion(t + h/2, X + h*K2/2, n, l, m)
        K4 = Funcion(t + h, X + h*K3, n, l, m) 
        X_new = X + h*(K1 + 2*K2 + 2*K3 + K4)/6
        V_calculado = np.insert(X, 0, t)
        Valores.append(V_calculado)
        X = X_new
    return np.array(Valores)

def RK5(Funcion, n, l, m, X0, t, t_max, h):  ## Metodo RK5, usar si es necesario
    X = np.array(X0)
    Valores = []
    for t in np.arange(t, t_max, h):
        K1 = Funcion(t, X, n, l, m)
        K2 = Funcion(t + h/4, X + h*K1/4, n, l, m)
        K3 = Funcion(t + h/4, X + h*K1/8 + h*K2/8, n, l, m)
        K4 = Funcion(t + h/2, X - h*K2/2 + h*K3, n, l, m)
        K5 = Funcion(t + 3*h/4, X + 3*K1*h/16 + 9*K4*h/16, n, l, m)
        K6 = Funcion(t + h, X - 3*K1*h/7 + 2*K2*h/7 + 12*K3*h/7 - 12*K4*h/7 + 8*K5*h/7, n, l, m)
        X_new = X + (7*K1 + 32*K3 + 12*K4 + 32*K5 + 7*K6)*h/90
        V_calculado = np.insert(X, 0, t)
        Valores.append(V_calculado)
        X = X_new
    return np.array(Valores)

def PenduloN(n, l, m, X, t, t_max, h, orden): ## Funcion que calcula numericamente el movimiento del N - pendulo
    if orden == 4:
        Res = RK4(Funcion, n, l, m, X, t, t_max, h)
    elif orden == 5:
        Res = RK5(Funcion, n, l, m, X, t, t_max, h)
    return Res

x1 = np.ones((n, 1))*(3*np.pi/4) ## Condiciones iniciales de los angulos 
v1 = np.ones((n, 1))*0 ## Condiciones iniciales de las velocidades angulares 
X1 = np.append(x1, v1)

x2 = np.ones((n, 1))*(3*np.pi/4 + 0.0001) ## Condiciones iniciales de los angulos con una perturbacion
v2 = np.ones((n, 1))*0 ## Condiciones iniciales de las velocidades angulares
X2 = np.append(x2, v2)

tmax = 30 ## Tiempo (en s) hasta donde sera realizado el calculo
Respuesta1 = PenduloN(n, l, m, X1, 0, tmax, 0.01, orden)
Respuesta2 = PenduloN(n, l, m, X2, 0, tmax, 0.01, orden)         

## Para guardar los datos
np.savetxt(f'{n}-P_1_RK{orden}.txt', Respuesta1) 
np.savetxt(f'{n}-P_2_RK{orden}.txt', Respuesta2)

