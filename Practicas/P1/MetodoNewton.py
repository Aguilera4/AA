# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: SERGIO AGUILERA RAMIREZ
"""

import numpy as np

print('EJERCICIO METODO DE NEWTON\n')
print('Ejercicio 2.1\n')

#Función
def F(u,v):
    return u**2 + 2*v**2 + 2*np.sin(2*u*np.pi)*np.sin(2*np.pi*v)

#Derivada parcial de F con respecto a u
def dFu(u,v):
    return 2*u + 4*np.pi*np.cos(2*np.pi*u)*np.sin(2*np.pi*v) #Derivada parcial de E con respecto a u
    
#Derivada parcial de F con respecto a v
def dFv(u,v):
    return 4*v + 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v)  #Derivada parcial de E con respecto a v

#Segundo derivada de u
def d2Fu(u,v):
    return 2-8*np.pi**2*np.sin(2*np.pi*v)*np.sin(2*np.pi*u)

#segunda derivada de v
def d2Fv(u,v):
    return 4 - 8*np.pi**2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v)

#Segunda derivada de F respecto a v
def dvu(u,v):
    return 8*np.pi**2*np.cos(2*np.pi*u)*np.cos(2*np.pi*v)

#Segunda derivada de F respecto a u
def duv(u,v):
    return 8*np.pi**2*np.cos(2*np.pi*v)*np.cos(2*np.pi*u)


#Gradiente de f
def gradF(u,v):
    return np.array([dFu(u,v), dFv(u,v)])

#
def metodo_newton(w,umb,eta,maxIter):
    salir = False
    it = 0.0
    d = []
    
    while it<maxIter and salir == False:
        q = F(w[0],w[1])
        
        #Matriz para representación en gráfica
        d.append([it, F(w[0],w[1])])
        
        #Matriz heussiana
        heussiana = np.array([[d2Fu(w[0],w[0]),duv(w[0],w[1])],[dvu(w[0],w[1]), d2Fv(w[0],w[1])]])
        
        #Coordenadas
        w = w - eta * np.dot(np.linalg.pinv(heussiana),gradF(w[0],w[1]))
        
        #Condición de salida
        if abs(F(w[0],w[1])-q) < umb:
            salir = True
        
        it+=1.0
        
    return w, d


####################### Ejercicio 2.1 METODO DE NEWTON ###############################################
#Learning rate
eta = 0.01 

#Valores iniciales
initial_point1 = np.array([0.1,0.1])
initial_point2 = np.array([1.0,1.0])
initial_point3 = np.array([-0.5,-0.5])
initial_point4 = np.array([-1.0,-1.0])

#Llamada al metodo de newton de los distintos valores iniciales
w1, d1 = metodo_newton(initial_point1,0.000001,0.01,200)
w2, d2 = metodo_newton(initial_point2,0.000001,0.01,200)
w3, d3 = metodo_newton(initial_point3,0.000001,0.01,200)
w4, d4 = metodo_newton(initial_point4,0.000001,0.01,200)

d1 = np.array(d1,np.float64)
d2 = np.array(d2,np.float64)
d3 = np.array(d3,np.float64)
d4 = np.array(d4,np.float64)

#Minimos
minimo1 = F(w1[0],w1[1])
minimo2 = F(w2[0],w2[1])
minimo3 = F(w3[0],w3[1])
minimo4 = F(w4[0],w4[1])

####################### TABLA COMPARATIVA #######################
print('TABLA COMPARATIVA\n')
print('Punto inicial            Coordenadas donde alcanza el mínimo                        Valor mínimo \n')
print(' (0.1,0.1)            (', w1[0], ', ', w1[1],')             (', minimo1, ')')
print(' (1,1)                (', w2[0], ' ,  ', w2[1],')              (', minimo2 ,')')
print(' (-0.5,-0.5)          (', w3[0], ', ', w3[1],')             (', minimo3 ,')')
print(' (-1,-1)              (', w4[0], ', ', w4[1],')              (', minimo4 ,')')
print('\n\n')
