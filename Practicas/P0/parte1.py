# -*- coding: utf-8 -*-

import numpy as np   #importo libreria numpy
import matplotlib.pyplot as plt #importo la libreria matplotlib
from sklearn import datasets #importo la database iris

iris = datasets.load_iris() #cargamos la database
x = iris.data #guardamos las caracteristicas
y = iris.target #guardamos las clases

array = x[:,2:] #guardamos las dos últimas caracteristicas

datos_clase_0 = array[0:49] #cojemos las 50 plantas de clase 0
datos_clase_1 = array[50:99] #cojemos las 50 plantas de clase 1
datos_clase_2 = array[100:149] #cojemos las 50 plantas de clase 2


plt.scatter(datos_clase_0[:,0:1],datos_clase_0[:,1:2], c ='r', label = 'Class_0') #dibujamos los puntos de la clase 0 (rojo)
plt.scatter(datos_clase_1[:,0:1],datos_clase_1[:,1:2], c ='b', label = 'Class_1') #dibujamos los puntos de la clase 1 (azul)
plt.scatter(datos_clase_2[:,0:1],datos_clase_2[:,1:2], c ='g', label = 'Class_2') #dibujamos los puntos de la clase 2 (verde)
plt.xlabel('I am x axis')
plt.ylabel('I am y axis')
plt.title('Parte 1')
plt.legend()
plt.axis([0,10,0,10]) #ajustamos los ejes
plt.show()

