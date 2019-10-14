# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: SERGIO AGUILERA RAMIREZ
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)

# simula_unif()
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

# simula_gaus()
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

# simula_recta()
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

# simulamos las nubes de puntos
e1 = simula_unif(50, 2, [-50,50])
e2 = simula_gaus(50, 2, np.array([5,7]))

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

# funcion que calcula la distancia respecto a la recta
def f(x, y, a, b):
	return signo(y - a*x - b)

# genera el conjunto de datos
N = simula_unif(100,2,[-50,50])
# simula la pendiente y el termino independiente de la recta en el rango[-50,50]
a,b = simula_recta([-50,50])


# dividimos los datos en positivos y negativos
datos_pos = []
datos_neg = []
etiq_pos = []
etiq_neg = []
for i in N:
    etiqueta = f(i[0], i[1], a, b)
    if etiqueta >= 1:
        datos_pos.append(np.array([i[0],i[1]]))
        etiq_pos.append(etiqueta)
    else:
        datos_neg.append(np.array([i[0],i[1]]))
        etiq_neg.append(etiqueta)

datos_pos = np.array(datos_pos, np.float64)
datos_neg = np.array(datos_neg, np.float64)

# concatenamos ambas clases
datos_orig = np.concatenate((datos_pos,datos_neg), axis=0)
etiquetas_orig = np.concatenate((etiq_pos,etiq_neg),axis=0)


# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

# función que introduce un 10% de ruido en las etiquetas
def ruido(x):
    num_cambio = round(len(x) * 10 / 100)

    for i in range(np.int64(num_cambio)):
        n = np.random.choice(len(x)-1)
        if x[n] >= 1: 
            x[n] = -1
        else:
            x[n] = 1
            
    return x

# introducimos ruido
etiq_pos = ruido(etiq_pos)
etiq_neg = ruido(etiq_neg)

# concatenamos los datos
datos_ruido = np.concatenate((datos_pos,datos_neg), axis=0)
etiquetas_ruido = np.concatenate((etiq_pos,etiq_neg),axis=0)

# dividimos el conjunto de datos en positivos y negativos
datos_pos2 = []
datos_neg2 = []

t=0
for i in datos_ruido:
    if etiquetas_ruido[t] >= 1:
        datos_pos2.append(np.array([i[0],i[1]]))
    else:
        datos_neg2.append(np.array([i[0],i[1]]))
    
    t+=1

datos_pos2 = np.array(datos_pos2, np.float64)
datos_neg2 = np.array(datos_neg2, np.float64)



###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON
print('\n\nEjericicio 2.1')

# algoritmo perceptron
def ajusta_PLA(datos, etiq, max_iter, vini):
    it = 0
    vini_ant = np.copy(vini)
    
    # iterar hasta superar un máximo de iteraciones
    while it < max_iter:
        it += 1

        # iteramos sobre todos las componentes del conjunto
        for i in range(len(datos)):
            # comprobamos si clasifica bien el elemento, si no lo clasifica de forma
            # correcta, actualizamos w
            if np.sign(np.dot(np.transpose(vini),datos[i])) != etiq[i]:
                vini = vini + etiq[i] * datos[i]
               
        # si w_actual es igual a w_anterior entonces paramos la ejecucion
        if np.array_equal(vini_ant,vini):
            break
        
        # copio w_actual en w_anterior
        vini_ant = np.copy(vini)
        
    return vini, it  

print('\n 1) Datos sin ruido\n')
print('a)')
    
# añadimos un termino a cada fila de la matriz con valor 1
datos_orig_aux = np.c_[np.ones(datos_orig.shape[0]), datos_orig]

## vini[0.0,0.0,0.0]
vini=[0.0,0.0,0.0]

# calculo el vector de pesos mediante el algoritmo de perceptron
w,it1 = ajusta_PLA(datos_orig_aux,etiquetas_orig,1000,vini)  

# calculamos la pendiente y el termino independiente de la recta
# respecto al vector de pesos
a = -(w[0]/w[2])/(w[0]/w[1])
b = -w[0]/w[2]

# generamos un conjunto de datos uniformes en el rango[-50,50],
# calculamos las respectivas 'y' del conjunto generado
x = np.linspace(-50,50,100)
y = a*x +b

#Gráfica
plt.axis([-60,60,-60,60])  
plt.plot(x,y, 'r--', label="recta a*x + b ")
plt.plot(x,y,'k--', label='a*x + b respecto w')
plt.scatter(datos_pos[:,0:1], datos_pos[:,1:2], c='b', label="positivos")
plt.scatter(datos_neg[:,0:1], datos_neg[:,1:2], c='g', label ="negativos")
plt.title('Ajuste PLA, vini[0,0,0]')
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.legend()
plt.show()
print("Numero de iterationes (vector inicial a 0): ", it1)

input("\n--- Pulsar tecla para continuar ---\n")

print('b)')
## Random initializations
iterations1 = []
for i in range(0,10):
    # genero el vector de pesos inicial de forma uniforme aleatoria
    vini = np.random.uniform(0,1,3)
    # calculo el vector de pesos mediante el algoritmo de perceptron
    w, it2 = ajusta_PLA(datos_orig_aux,etiquetas_orig,1000,vini)
    iterations1.append(it2)
    plt.axis([-60,60,-60,60])
    # calculamos la pendiente y el termino independiente de la recta
    # respecto al vector de pesos
    a = -(w[0]/w[2])/(w[0]/w[1])
    b = -w[0]/w[2]
    
    # generamos un conjunto de datos uniformes en el rango[-50,50],
    # calculamos las respectivas 'y' del conjunto generado
    x = np.linspace(-50,50,100)
    y = a*x +b
    # si es la primera iteracion pintamos la leyenda, para que no salga repetida 10 veces
    if i == 0:
        plt.plot(x,y, 'k--',label='a*x + b respecto w')
    else:
        plt.plot(x,y, 'k--')

# Gráfica
plt.plot(x,y, 'r--', label="recta a*x + b ")
plt.scatter(datos_pos[:,0:1], datos_pos[:,1:2], c='b', label="positivos")
plt.scatter(datos_neg[:,0:1], datos_neg[:,1:2], c='g', label ="negativos")
plt.title('Ajuste PLA, vini[random]')
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.legend()
plt.show()
    
print('Valor medio de iteraciones necesario para converger (vector inicial aleatorio): {}'.format(np.mean(np.asarray(iterations1))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b
print('\n 2) Datos con ruido\n')
print('a)')

# añadimos un termino a cada fila de la matriz con valor 1
datos_ruido_aux = np.c_[np.ones(datos_ruido.shape[0]), datos_ruido]

## vini[0.0,0.0,0.0]
vini=[0.0,0.0,0.0]

# calculo el vector de pesos mediante el algoritmo de perceptron
w2,it3 = ajusta_PLA(datos_ruido_aux,etiquetas_ruido,1000,vini) 
 
# calculamos la pendiente y el termino independiente de la recta
# respecto al vector de pesos
a = -(w[0]/w[2])/(w[0]/w[1])
b = -w[0]/w[2]

# generamos un conjunto de datos uniformes en el rango[-50,50],
# calculamos las respectivas 'y' del conjunto generado
x = np.linspace(-50,50,50)
y = a*x +b

# Gráfica
plt.axis([-60,60,-60,60])  
plt.plot(x,y, 'r--', label="recta a*x + b ")
plt.plot(x,y,'k--',label='a*x + b respecto w')
plt.scatter(datos_pos2[:,0:1], datos_pos2[:,1:2], c='b', label="positivos")
plt.scatter(datos_neg2[:,0:1], datos_neg2[:,1:2], c='g', label ="negativos")
plt.title('Ajuste PLA, vini[0,0,0]')
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.legend()
plt.show()
print("Numero de iterationes (vector inicial a 0): ", it3)

input("\n--- Pulsar tecla para continuar ---\n")

print('b)')
# Random initializations
iterations2 = []
for i in range(0,10):
    # genero el vector de pesos inicial de forma uniforme aleatoria
    vini = np.random.uniform(0,1,3)
    # calculo el vector de pesos mediante el algoritmo de perceptron
    w, it4 = ajusta_PLA(datos_orig_aux,etiquetas_ruido,1000,vini)
    iterations2.append(it4)
    # calculamos la pendiente y el termino independiente de la recta
    # respecto al vector de pesos
    a = -(w[0]/w[2])/(w[0]/w[1])
    b = -w[0]/w[2]
    # generamos un conjunto de datos uniformes en el rango[-50,50],
    # calculamos las respectivas 'y' del conjunto generado
    x = np.linspace(-50,50,50)
    y = a*x +b
    # si es la primera iteracion pintamos la leyenda, para que no salga repetida 10 veces
    if i == 0:
        plt.plot(x,y, 'k--',label='a*x + b respecto w')
    else:
        plt.plot(x,y, 'k--')

# Gráfica
plt.plot(x,y, 'r--', label="recta a*x + b ")
plt.scatter(datos_pos2[:,0:1], datos_pos2[:,1:2], c='b', label="positivos")
plt.scatter(datos_neg2[:,0:1], datos_neg2[:,1:2], c='g', label ="negativos")
plt.title('Ajuste PLA, vini[random]')
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.legend()
plt.show()
    
    
print('Valor medio de iteraciones necesario para converger (vector inicial aleatorio): {}'.format(np.mean(np.asarray(iterations2))))
  
input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
print('\n\nEjericicio 2.2')

# algoritmo de regresión logística
def sgdRL(datos, etiquetas, lr, dif_l, num_epocas):
    # vector inicial a 0.0
    w = w_ant = [0.0,0.0,0.0]
    # generamos la matriz con los datos y sus etiquetas
    matriz = np.c_[datos,etiquetas]
    
    dif = 9999.0
    epoca = 0
    
    # mientras no se supero el número máximo de epocas
    while  epoca < num_epocas:
        epoca +=1
        # aleatorizamos la matriz
        np.random.shuffle(matriz)
        # guardamos las etiquetas
        y = matriz[:,3]
        #guardamos los datos
        x = matriz[:,0:3]
        # copiamos el vector de pesos actual
        w_ant = np.copy(w)
        
        # iteramos en minibatches de 32
        for i in range(0, datos.shape[0], 32):
            # escogemos los datos de 32 en 32 elementos
            x_grad = x[i:i+32,:]
            y_grad = y[i:i+32]
            # establecemos el gradiente a 0.0
            gradiente = [0.0,0.0,0.0]
            # iteramos sobre los elementos del minibatches
            for j in range(len(y_grad)):
                # calculamos el gradiente
                gradiente +=  np.dot(np.dot(-y_grad[j],x_grad[j]),np.exp(np.dot(-y_grad[j],np.transpose(w))@x_grad[j]))
            
            #gradiente = np.dot((1.0/len(y_grad)) , gradiente)
            # actualizamos el vector de pesos
            w = w - lr * gradiente  
      
        # calculamos la diferencia entre w_anterior y w_actual
        dif = np.linalg.norm(w_ant - w)
        
        # si la diferencia es menor que epsilon (0.01) paramos la ejecución
        if dif < dif_l:
            break
        
    return w


# funcion para el calculo del error
def Err(datos, etiquetas,w):
    suma = 0
    for i in range(len(datos)):
        suma += np.log(1 + np.exp(-etiquetas[i] * np.dot(np.transpose(w),datos[i] )))
        
    return suma / len(datos)


# simulamos el conjunto de datos
x2 = simula_unif(100,2,[0,2])
# simulamos la recta en el rango [0,2]
a,b = simula_recta([0,2])


# dividimos los datos
etiq = []
positivos = []
negativos = []
for i in x2:
    etiqueta = f(i[0], i[1], a, b)    
    etiq.append(etiqueta)
    
    if etiqueta == 1.0:
        positivos.append(np.array([i[0],i[1]]))
    if etiqueta == -1.0:
        negativos.append(np.array([i[0],i[1]]))

positivos = np.array(positivos, np.float64)
negativos = np.array(negativos, np.float64)    


## añado 1 en el primer termino
x2_aux = np.c_[np.ones(x2.shape[0]), x2]

## llamada a la funcion de regresión logística
w = sgdRL(x2_aux,etiq,0.01,0.01,3000)

# generamos un conjunto de datos uniformes en el rango[-50,50],
# calculamos las respectivas 'y' del conjunto generado
x = np.linspace(0,2,100)
y = a*x + b

# calculamos la pendiente y el termino independiente de la recta
# respecto al vector de pesos
a1 = -(w[0]/w[2])/(w[0]/w[1])
b1 = -w[0]/w[2]
x1 = np.linspace(0,2,100)
# generamos un conjunto de datos uniformes en el rango[-50,50],
# calculamos las respectivas 'y' del conjunto generado
y1 = a1*x1 + b1

# Gráfica
plt.plot(x,y, c="r", label="recta 0/1")
plt.plot(x1,y1, c="k", label="recta con vector de pesos w")
plt.axis([0,2,0,2])  
plt.scatter(positivos[:,0:1], positivos[:,1:2], c='b', label="positivos")
plt.scatter(negativos[:,0:1], negativos[:,1:2], c='g', label ="negativos")
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.legend()
plt.title('RL')
plt.show()
    
# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

# error de entrada
print("Ein: ", Err(x2_aux,etiq,w))

# calculamos los errores para 1000 muestras
errores = []
for i in range(1000):
    datos = simula_unif(100,2,[0,2])

    etiq = []
    for i in datos:
        etiqueta = f(i[0], i[1], a, b)    
        etiq.append(etiqueta)
    
    datos_aux = np.c_[np.ones(datos.shape[0]), datos]
    errores.append(Err(datos_aux,etiq,w))

# media de los errores obtenidos
print("Error medio de las 1000 muestras :" , sum(errores)/1000)
    
    