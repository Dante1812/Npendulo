#!/usr/bin/python3
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import time

n = 1 ## Valor de N
g = 9.81 ## Valor de g
orden = 4 ## Orden para el metodo de Runge - Kutta

m = np.ones(n) ## Valores para las masas
l = np.ones(n) ## Valores para las longitudes de la cuerda

def Funcion(t, X): ## Funcion que calcula las 2n EDO's de primer orden
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
                B[i][j] *= g*np.sin(X[i])
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

def RK4(Funcion, X0, t, t_max, h): ## Metodo RK4
    X = np.array(X0)
    Valores = []
    for t in np.arange(t, t_max, h):
        K1 = Funcion(t, X)
        K2 = Funcion(t + h/2, X + h*K1/2)
        K3 = Funcion(t + h/2, X + h*K2/2)
        K4 = Funcion(t + h, X + h*K3) 
        X_new = X + h*(K1 + 2*K2 + 2*K3 + K4)/6
        V_calculado = np.insert(X, 0, t)
        Valores.append(V_calculado)
        X = X_new
    return np.array(Valores).T

def RK5(Funcion, X0, t, t_max, h):  ## Metodo RK5, usar si es necesario
    X = np.array(X0)
    Valores = []
    for t in np.arange(t, t_max, h):
        K1 = Funcion(t, X)
        K2 = Funcion(t + h/4, X + h*K1/4)
        K3 = Funcion(t + h/4, X + h*K1/8 + h*K2/8)
        K4 = Funcion(t + h/2, X - h*K2/2 + h*K3)
        K5 = Funcion(t + 3*h/4, X + 3*K1*h/16 + 9*K4*h/16)
        K6 = Funcion(t + h, X - 3*K1*h/7 + 2*K2*h/7 + 12*K3*h/7 - 12*K4*h/7 + 8*K5*h/7)
        X_new = X + (7*K1 + 32*K3 + 12*K4 + 32*K5 + 7*K6)*h/90
        V_calculado = np.insert(X, 0, t)
        Valores.append(V_calculado)
        X = X_new
    return np.array(Valores).T

def RK8(Funcion, X0, t, t_max, h):  ## Usar en caso de que RK5 falle :v
    X = np.array(X0)
    Valores = []
    for t in np.arange(t, t_max, h):
        K1 = Funcion(t, X)
        K2 = Funcion(t + h/18, X + h*K1/18)
        K3 = Funcion(t + h/12, X + h*K1/48 + h*K2/16)
        K4 = Funcion(t + h/8, X + h*K1/32 + 3*h*K3/32)
        K5 = Funcion(t + 5*h/16, X + 5*h*K1/16 - 75*h*K3/64 + 75*h*K4/64)
        K6 = Funcion(t + 3*h/8, X + 3*h*K1/80 + 3*h*K4/16 + 3*h*K5/20)
        K7 = Funcion(t + 59*h/400, X + 29443841*h*K1/614563906 + 77736538*h*K4/692538347 - 28693883*h*K5/1125000000 + 23124283*h*K6/1800000000)
        K8 = Funcion(t + 93*h/200, X + 16016141*h*K1/946692911 + 61564180*h*K4/158732637 + 22789713*h*K5/633445777 + 545815736*h*K6/2771057229 - 180193667*h*K7/1043307555)
        K9 = Funcion(t + 5490023248*h/9719169821, X + 39632708*h*K1/573591083 - 433636366*h*K4/683701615 - 421739975*h*K5/2616292301 + 100302831*h*K6/723423059 + 790204164*h*K7/839813087 + 800635310*h*K8/3783071287)
        K10 = Funcion(t + 13*h/20, X + 246121993*h*K1/1340847787 - 37695042795*h*K4/15268766246 - 309121744*h*K5/1061227803 - 12992083*h*K6/490766935 + 6005943493*h*K7/2108947869 + 393006217*h*K8/1396673457 + 123872331*h*K9/1001029789)
        K11 = Funcion(t + 1201146811*h/1299019798, X - 1028468189*h*K1/846180014 + 8478235783*h*K4/508512852 + 1311729485*h*K5/1432422823 - 10304129995*h*K6/1701304382 - 48777925059*h*K7/3047939560 + 15336726248*h*K8/1032824649 - 45442868181*h*K9/3398467696 + 3065993473*h*K10/597172653)
        K12 = Funcion(t + h, X + 185892177*h*K1/718116043 - 3185094517*h*K4/667107341 - 477755414*h*K5/1098053517 - 703635378*h*K6/230739211 + 5731566787*h*K7/1027545527 + 5232866602*h*K8/850066563 - 4093664535*h*K9/808688257 + 3962137247*h*K10/1805957418 + 65686358*h*K11/487910083)
        K13 = Funcion(t + h, X + 403863854*h*K1/491063109 - 5068492393*h*K4/434740067 - 411421997*h*K5/543043805 + 652783627*h*K6/914296604 + 11173962825*h*K7/925320556 - 13158990841*h*K8/6184727034 + 3936647629*h*K9/1978049680 - 160528059*h*K10/685178525 + 248638103*h*K11/1413531060)
        X = X + h*(14005451*K1/335480064 - 59238493*K6/1068277825 + 181606767*K7/758867731 + 561292985*K8/797845732 - 1041891430*K9/1371343529 + 760417239*K10/1151165299 + 118820643*K11/751138087 - 528747749*K12/2220607170 + K13/4)
        V_calculado = np.insert(X, 0, t)
        Valores.append(V_calculado)
    return np.array(Valores).T

def PenduloN(Funcion, X, t, t_max, h, orden): ## Funcion que calcula numericamente el movimiento del N - pendulo
    if orden == 4:
        Res = RK4(Funcion, X, t, t_max, h)
    elif orden == 5:
        Res = RK5(Funcion, X, t, t_max, h)
    elif orden == 8:
        Res = RK8(Funcion, X, t, t_max, h)
    return Res

x1 = np.ones((n, 1))*(np.pi*3/4) ## Condiciones iniciales de los angulos 
v1 = np.ones((n, 1))*0 ## Condiciones iniciales de las velocidades angulares 
X1 = np.append(x1, v1)

x2 = np.ones((n, 1))*(np.pi*3/4 + 1e-10) ## Condiciones iniciales de los angulos con una perturbacion
v2 = np.ones((n, 1))*0 ## Condiciones iniciales de las velocidades angulares
X2 = np.append(x2, v2)

tmax = 10000 ## Tiempo (en s) hasta donde sera realizado el calculo
Respuesta1 = PenduloN(Funcion, X1, 0, tmax, 0.01, orden)
Respuesta2 = PenduloN(Funcion, X2, 0, tmax, 0.01, orden)         

## Para guardar los datos
#np.savetxt(f'{n}-P_1_RK{orden}_prueba.txt', Respuesta1) 
#np.savetxt(f'{n}-P_2_RK{orden}_prueba.txt', Respuesta2)
np.savetxt(f'{n}-P_1_RK{orden}.txt', Respuesta1) 
np.savetxt(f'{n}-P_2_RK{orden}.txt', Respuesta2)

#Time = []
#for _ in range(10):
#    t0 = time.time()
#    Respuesta1 = PenduloN(Funcion, X1, 0, tmax, 0.01, orden)
#    t1 = time.time()
#    Time.append(t1-t0)
#print(f'Tiempo de ejecucion: {np.mean(Time)}')

