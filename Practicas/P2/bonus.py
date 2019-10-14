# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: SERGIO AGUILERA RAMIREZ
"""


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='green', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='green', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

# función para el calculo del error
def Err(x,y,w):
    return sum(np.sign(np.dot(x,w.T)) != y)/len(y)

print("\n1) Regresion lineal\n")
#LINEAR REGRESSION FOR CLASSIFICATION 

# pseudoinversa
def pseudoinverse(datos,etiquetas):
    w = np.linalg.pinv(datos)
    w = np.dot(w,etiquetas)
    return w

w = pseudoinverse(x,y)

# Errores
Ein = Err(x,y,w)
Eout = Err(x_test,y_test,w)

print("Ein (Pseudoinversa): ", Ein)
print("Eout (Pseudoinversa): ", Eout)
    
input("\n--- Pulsar tecla para continuar ---\n")


print("\n\n2) PLA-Pocket\n")

# POCKET ALGORITHM
def pocket_algorithm(x,y,vini,max_it,difm):
    w = w_final = w_ant = vini
    it = 0
    error_min = 9999.0

    
    while it < max_it:
        
        for i in range(len(y)):
            if np.sign(np.dot(np.transpose(w),x[i])) != y[i]:
                w = w + y[i] * x[i]
                
        if (Err(x,y,w) < error_min):
            error_min = Err(x,y,w)
            w_final = w
        
        dif = np.linalg.norm(w_ant - w)     
        if dif < difm:
            break
    
        
        it += 1
        w_ant = np.copy(w)
            
    return w_final
    

w1 = pocket_algorithm(x,y,w,1000,0.01)

# errores del algoritmo pocket
Ein = Err(x,y,w1)
Eout = Err(x_test,y_test,w1)

print("Ein (PLA-Pocket): ", Ein)
print("Eout (PLA-Pocket): ", Eout)


input("\n--- Pulsar tecla para continuar ---\n")

print("GRAFICAS\n")

## gráfica de la pseudoinversa con datos entrenamiento
fig, ax = plt.subplots()
a1 = -(w[0]/w[2])/(w[0]/w[1])
b1 = -w[0]/w[2]
x1 = np.linspace(0,0.5,100)
y1 = a1*x1 + b1
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='green', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
plt.plot(x1,y1, "k", label="a*x+b")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='REGRESION LINEAR (PSEUDOINVERSE) + datos entrenamiento')
ax.set_xlim((0, 1))
plt.legend()
plt.show()


# grafica de la pseudoinversa con datos test
fig, ax = plt.subplots()
a2 = -(w[0]/w[2])/(w[0]/w[1])
b2 = -w[0]/w[2]
x2 = np.linspace(0,0.5,100)
y2 = a2*x2 + b2
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='green', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
plt.plot(x2,y2, "k", label="a*x+b")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='REGRESION LINEAR (PSEUDOINVERSE) + datos_test')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

# grafica del algoritmo pocket con datos entrenamiento
fig, ax = plt.subplots()
a3 = -(w[0]/w[2])/(w[0]/w[1])
b3 = -w[0]/w[2]
x3 = np.linspace(0,0.5,100)
y3 = a3*x3 + b3
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='green', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
plt.plot(x3,y3, "k", label="a*x+b")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Pocket + datos entrenamiento')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

# grafica del algoritmo pocket con datos test
fig, ax = plt.subplots()
a4 = -(w[0]/w[2])/(w[0]/w[1])
b4 = -w[0]/w[2]
x4 = np.linspace(0,0.5,100)
y4 = a4*x4 + b4
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='green', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
plt.plot(x4,y4, "k", label="a*x+b")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Pocket + datos test')
ax.set_xlim((0, 1))
plt.legend()
plt.show()        


input("\n--- Pulsar tecla para continuar ---\n")

print("COTAS\n")
#COTA SOBRE EL ERROR
def calcula_cota(err, tam, dim, tolerancia ):
    return err + np.sqrt((8/tam)*np.log((4*((2*tam)**dim + 1)) / tolerancia))

## Acotacion de los errores
cota_Ein = calcula_cota(Ein,x.shape[0],3,0.05)
cota_Eout = calcula_cota(Eout,x_test.shape[0],3,0.05)

print("Cota de Ein: ", cota_Ein)
print("Cota de Eout: ", cota_Eout)


