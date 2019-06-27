# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: SERGIO AGUILERA RAMIREZ
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

#Función
def E(u,v):
    return (u**2*np.e**v-2*v**2*np.e**-u)**2  #function   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**2*np.e**v-2*v**2*np.e**-u)*(2*u*np.e**v+2*np.e**-u*v**2) #Derivada parcial de E con respecto a u
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**2*np.e**v-2*v**2*np.e**-u)*(u**2*np.e**v-4*np.e**-u*v)  #Derivada parcial de E con respecto a v

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#Gradiente descendente
def gradient_descent():	
	iterations = 0
	salir = False
	w = initial_point

	while iterations < maxIter and salir == False:
		w = w - eta*gradE(w[0],w[1])
		salir = E(w[0],w[1]) < error2get
		iterations = iterations + 1
	return w, iterations    

eta = 0.01 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent()

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Error: ', E(w[0],w[1]))

# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...



####################### EJERCICIO 1.3 ###############################################

#Función
def F(u,v):
    return u**2 + 2*v**2 + 2*np.sin(2*u*np.pi)*np.sin(2*np.pi*v)

#Derivada parcial de F con respecto a u
def dFu(u,v):
    return 2*u + 4*np.pi*np.cos(2*np.pi*u)*np.sin(2*np.pi*v) #Derivada parcial de E con respecto a u
    
#Derivada parcial de F con respecto a v
def dFv(u,v):
    return 4*v + 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v)  #Derivada parcial de E con respecto a v

#Gradiente de f
def gradF(u,v):
    return np.array([dFu(u,v), dFv(u,v)])

#Gradiente descendente de F
def gradient_descentF():  
    iterations = 0
    w = initial_point
    min = F(w[0],w[1])
    valores = [min]
    iteraciones = [0]

    while iterations < 50:
         w =  w - eta*gradF(w[0],w[1])
         iterations += 1
         min = F(w[0],w[1])
         iteraciones.append(iterations)
         valores.append(min)
    return w, iterations, valores, iteraciones


####################### Ejercicio 1.3 (0.1,0.1) Learning rate = 0.01 ###############################################
#Caculamos el mínimo para el punto inicial (0.1,0.1) y un learning rate de 0.01
eta = 0.01 
initial_point = np.array([0.1,0.1])
w1, it1, valores, ejeX = gradient_descentF()
minimo1 = F(w1[0],w1[1])
print ('Punto inicial (0.1,0.1)')
print ('Numero de iteraciones: ', it1)
print ('Coordenadas obtenidas: (', w1[0], ', ', w1[1],')')
print ('Mínimo: (', minimo1, ')')

# GRAPHICS
plt.plot(ejeX,valores)
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('1.3 -- Initial_point(0.1,0.1) Learning rate = 0.01')
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")   

####################### Ejercicio 1.3 (0.1,0.1) Learning rate = 0.1 ###############################################
#Caculamos el mínimo para el punto inicial (0.1,0.1) y un learning rate de 0.1
eta = 0.1 
initial_point = np.array([0.1,0.1])
w11, it11, valores11, ejeX11 = gradient_descentF()
minimo11 = F(w11[0],w11[1])

print ('Punto inicial (0.1,0.1)')
print ('Numero de iteraciones: ', it11)
print ('Coordenadas obtenidas: (', w11[0], ', ', w11[1],')')
print ('Mínimo: (', minimo11, ')')

# GRAPHICS
plt.plot(ejeX11, valores11)
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('1.3 -- Initial_point(0.1,0.1) Learning rate = 0.1')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")   

####################### Ejercicio 1.3 (1,1) ###############################################
#Caculamos el mínimo para el punto inicial (1,1)
eta = 0.01 
initial_point = np.array([1,1])
w2, it2, valores2, ejeX2 = gradient_descentF()
minimo2 = F(w2[0],w2[1])

####################### Ejercicio 1.3 (-0.5,-0.5) ###############################################
#Caculamos el mínimo para el punto inicial (-0.5,-0.5)
eta = 0.01 
initial_point = np.array([-0.5,-0.5])
w3, it3, valores3, ejeX3 = gradient_descentF()
minimo3 = F(w3[0],w3[1])

####################### Ejercicio 1.3 (-1,-1) ###############################################
#Caculamos el mínimo para el punto inicial (-1,-1)
eta = 0.01 
initial_point = np.array([-1,-1])
w4, it4, valores4, ejeX4 = gradient_descentF()
minimo4 = F(w4[0],w4[1])

####################### TABLA COMPARATIVA #######################
print('TABLA COMPARATIVA\n')
print('Punto inicial            Coordenadas donde alcanza el mínimo                        Valor mínimo \n')
print(' (0.1,0.1)            (', w1[0], ', ', w1[1],')             (', minimo1, ')')
print(' (1,1)                (', w2[0], ' ,  ', w2[1],')              (', minimo2 ,')')
print(' (-0.5,-0.5)          (', w3[0], ', ', w3[1],')             (', minimo3 ,')')
print(' (-1,-1)              (', w4[0], ', ', w4[1],')              (', minimo4 ,')')
print('\n\n')