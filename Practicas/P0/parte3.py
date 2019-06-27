# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

random_array = np.linspace(0,2*np.pi,100) #generamos el array con 100 elementos aleatorios entre 0 y 2*PI

array_sin = np.sin(random_array) #generamos el seno de todos los elementos anteriores
array_cos = np.cos(random_array) #generamos el coseno de todos los elementos anteriores
array_sincos = array_sin + array_cos # generamos el array de la suma de array_sin y array_cos

plt.plot(array_sin,'k--', label='array_sin') #curva del seno (negro)
plt.plot(array_cos,'b--', label='array_cos') #curva del coseno (azul)
plt.plot(array_sincos,'r--', label='array_sin+cos') #curva de la suma (rojo)
plt.xlabel('I am x axis')
plt.ylabel('I am y axis')
plt.title('Parte 3')
plt.legend()
plt.show()
