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


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
print("\nEjercicio 1.1\n")
print('a)')

# nube de puntos generada con simula_unif()
print('Simula_unif()')
x = simula_unif(50, 2, [-50,50])

# GRÁFICAS
plt.scatter(x[:,0:1], x[:,1:2], c='b')
plt.axis([-50,50,-50,50])  
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.title("simula_unif()")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('b)')

# nube de puntos generada con simula_gaus()    
print('\nSimula_gaus()')
x = simula_gaus(50, 2, np.array([5,7]))

# GRAPHICS
plt.scatter(x[:,0:1], x[:,1:2], c='g')
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.title("simula_gaus()")
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
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

    
print('\n\nEjericicio 1.2')
print('\na)')

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

# generamos un conjunto de datos uniformes en el rango[-50,50],
# calculamos las respectivas 'y' del conjunto generado
x = np.linspace(-50,50,100)
y = a*x + b

# Gráfica
plt.plot(x,y, 'k', label="a*x+b")
plt.scatter(datos_pos[:,0:1], datos_pos[:,1:2], c='b', label="positivos")
plt.scatter(datos_neg[:,0:1], datos_neg[:,1:2], c='g', label ="negativos")
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.legend()
plt.title("EJERCICIO 1.2 a")
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido


print('\nb)')

# función que introduce un 10% de ruido en las etiquetas
def ruido(x):
    num_cambio = round(len(x) * 10 / 100)
    n = np.random.randint(0,len(x)-1,num_cambio)   
    for i in range(np.int64(num_cambio)):
        t = n[i]
        if x[t] == 1: 
            x[t] = -1
        else:
            x[t] = 1
    return x

# introducimos ruido
etiq_pos = ruido(etiq_pos)
etiq_neg = ruido(etiq_neg)

# concatenamos los datos
datos = np.concatenate((datos_pos,datos_neg), axis=0)
etiquetas = np.concatenate((etiq_pos,etiq_neg),axis=0)


# dividimos el conjunto de datos en positivos y negativos
datos_pos2 = []
datos_neg2 = []
t=0
for i in datos:
    if etiquetas[t] >= 1:
        datos_pos2.append(np.array([i[0],i[1]]))
    else:
        datos_neg2.append(np.array([i[0],i[1]]))
    t+=1
    
datos_pos2 = np.array(datos_pos2, np.float64)
datos_neg2 = np.array(datos_neg2, np.float64)

# generamos un conjunto de datos uniformes en el rango[-50,50],
# calculamos las respectivas 'y' del conjunto generado
x = np.linspace(-50,50,100)
y = a*x + b        
    

# Gráfica
plt.plot(x,y, 'k', label="a*x+b")
plt.scatter(datos_pos2[:,0:1], datos_pos2[:,1:2], c='b', label="positivos")
plt.scatter(datos_neg2[:,0:1], datos_neg2[:,1:2], c='g', label="negativos")
plt.xlabel("EJE X")
plt.ylabel("EJE Y")
plt.legend()
plt.title("EJERCICIO 1.2 a")
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

print('\n\nEjericicio 1.3')

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=200, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
    
# definimos las diferentes funciones   
def f1(x):
    return (x[:,0] - 10)**2 + (x[:,1] - 20)**2 - 400
    
def f2(x):
    return 0.5 * (x[:,0] + 10)**2 + (x[:,1] - 20)**2 - 400
    
def f3(x):
    return 0.5 * (x[:,0] - 10)**2 - (x[:,1] + 20)**2 - 400
    
def f4(x):
    return x[:,1] - 20 * x[:,0]**2 - 5 * x[:,0] + 3
    
## f1
plot_datos_cuad(datos,etiquetas,f1,title="Función f1 = (x[:,0] - 10)**2 + (y[:,1] - 20)**2 - 400")

## f2
plot_datos_cuad(datos,etiquetas,f2,title="Función f2 = 0.5 * (x[:,0] + 10)**2 + (y[:,1] - 20)**2 - 400")

## f3
plot_datos_cuad(datos,etiquetas,f3,title="Función f3 = 0.5 * (x[:,0] - 10)**2 - (y[:,1] + 20)**2 - 400")

# f4
plot_datos_cuad(datos,etiquetas,f4,title="Función f4 = y[:,1] - 20 * x[:,0]**2 - 5 * y[:,0] + 3")