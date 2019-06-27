# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: SERGIO AGUILERA RAMIREZ
"""

import numpy as np
import matplotlib.pyplot as plt

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2\n')

label5 = 1
label1 = -1

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    j=0
    error = 0
    N=x.shape[0]
    for i in x:
      error += (np.transpose(w).dot(i)-y[j])**2
      j+=1
    
    error = error * (1/N)
    return error


# Gradiente Descendente Estocastico
def sgd(x,y,eta,umbral):
    j = 0
    maxminibach = round(y.size/64)
    w = [0.0,0.0,0.0]

    while j<100 and Err(x,y,w)<umbral and j < maxminibach:
        
        index = np.random.randint(0,x.shape[0], size = 64)
        np.random.shuffle(index)   
        data = x[index]
        etiq = y[index]
        
        p=0
        for i in data:
            for t in range(3):
                sumatoria = np.dot(i[t],(np.dot(w,i) - etiq[p]))
                w[t] = w[t] - eta * sumatoria    
            p+=1     
        
        j+=1
                    
            
    return w

# Pseudoinversa	
def pseudoinverse(datosEntrenamiento, label):
    w = np.linalg.pinv(datosEntrenamiento)
    w = np.dot(w,label)
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


######### DIVISION DE CLASES #########

data_label1 = []
data_label5 = []

j = 0

for i in y:
    if i == -1:
        data_label1.append(np.array([x[j][1],x[j][2]]))
    else:
        data_label5.append(np.array([x[j][1],x[j][2]]))
    j +=1

data_label1 = np.array(data_label1, np.float64)
data_label5 = np.array(data_label5, np.float64)


######### PSEUDOINVERSE #########
wp = pseudoinverse(x, y)

######### SGD #########
ws = sgd(x,y,0.01,1)

# GRAPHICS
plt.scatter(data_label1[:,0:1], data_label1[:,1:2], c='r',s=2)
plt.scatter(data_label5[:,0:1], data_label5[:,1:2], c='b',s=2)
m = np.amax(x)
t = np.arange(0.,m+0.5,0.5)
plt.plot(t, -wp[0]/wp[2]-wp[1]/wp[2]*t,'k', label = 'Pseudoinverse')
plt.plot(t, -ws[0]/ws[2]-ws[1]/ws[2]*t,'green', label = 'SGD')
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('PSEUDOINVERSA')
plt.legend()
plt.show()

print ('Bondad del resultado para pseudoinversa:')
print ("Ein: ", Err(x,y,wp))
print ("Eout: ", Err(x_test, y_test, wp))

print ('\nBondad del resultado para grad. descendente estocastico:')
print ("Ein: ", Err(x,y,ws))
print ("Eout: ", Err(x_test, y_test, ws))

input("\n--- Pulsar tecla para continuar ---\n")


######### EJERCICIO 2.2 (A) #########
print('\nEJERCICIO 2.2 (A)')
N = simula_unif(1000,2,1)
plt.scatter(N[:,0:1], N[:,1:2], c='r',s=2)
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('EJERCICIO 2.2 (A)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

######### EJERCICIO 2.2 (B) #########
print('\nEJERCICIO 2.2 (B)')
def asigna_etiqueta(x):
    etiquetas = []    
    for i in x:
        etiquetas.append(np.sign((i[0]-0.2)**2+i[1]**2-0.6))
        
    etiquetas = np.array(etiquetas, np.float64)
    index = np.random.randint(0,x.shape[0],size=100)
    np.random.shuffle(index)    
    etiquetas[index][0] = -etiquetas[index][0]

    return etiquetas

etiquetas = asigna_etiqueta(N)

d_label1 = []
d_label5 = []

j = 0
for i in etiquetas:
    if i == -1:
        d_label1.append(np.array([N[j][0],N[j][1]]))
    else:
        d_label5.append(np.array([N[j][0],N[j][1]]))
    j +=1
    
d_label1 = np.array(d_label1, np.float64)
d_label5 = np.array(d_label5, np.float64)

plt.scatter(d_label1[:,0:1], d_label1[:,1:2], c='r',s=2)
plt.scatter(d_label5[:,0:1], d_label5[:,1:2], c='b',s=2)
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('EJERCICIO 2.2 (B)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


######### EJERCICIO 2.2 (C) #########
print('\nEJERCICIO 2.2 (C)')

#Añadimos un 1 en la primera columna
d = np.ones((N.shape[0],3))
d[:,1:2] = N[:,0:1]
d[:,2:3] = N[:,1:2]
    
#Llamada a la funcion sgd
d1 = sgd(d,etiquetas,0.01,10) ## Llamada a la funcion de gradientes descendente estocastico

print ("Ein: ", Err(d,etiquetas,d1))

input("\n--- Pulsar tecla para continuar ---\n")

######### EJERCICIO 2.2 (D) #########
print('\n\nEJERCICIO 2.2 (D)\n')
t=0
suma_Ein = 0
suma_Eout = 0

while t<1000:
    
    #TRAIN
    n = simula_unif(1000,2,1)
    etiquetas = asigna_etiqueta(n)
    
    #Creo la matriz añadiendo 1 en la primera columna q = [[1 ...  ...] [1 ... ...]]
    q = np.ones((n.shape[0],3)) 
    q[:,1:2] = n[:,0:1]
    q[:,2:3] = n[:,1:2] 
    q = np.array(q, np.float64)

    #Llamada a la funcion sgd
    r = sgd(q,etiquetas,0.01,1.1) ## Llamada a la funcion de gradientes descendente estocastico
    
    #TEST
    u = simula_unif(1000,2,1)
    etiq2 = asigna_etiqueta(u)
    
    #Creo la matriz añadiendo 1 en la primera columna q = [[1 ...  ...] [1 ... ...]]
    e = np.ones((n.shape[0],3))
    e[:,1:2] = u[:,0:1]
    e[:,2:3] = u[:,1:2]
    e = np.array(e, np.float64)
    
    #Sumatoria de los Errores
    suma_Ein += Err(q,etiquetas,r)
    suma_Eout += Err(e, etiq2, r)
    t+=1
    
#Errores medios
Ein_medio = suma_Ein/1000
Eout_medio = suma_Eout/1000

print('Ein medio: ', Ein_medio)
print('Eout medio: ', Eout_medio)     
