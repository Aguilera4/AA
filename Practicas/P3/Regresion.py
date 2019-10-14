# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:48:52 2019

@author: SERGIO
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model, metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV , train_test_split
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

#------------------------------ FUNCIONES ------------------------------

# Funcion que genera las particones del conjunto de datos
def partitions(datos):
    x_train, y_train, x_test, y_test = train_test_split(datos)

# Función de lectura de los datos: lee los datos del fichero y divide en datos y etiquetas
def readData(rute):	
    data = pd.read_csv(rute,header=-1)
    data = np.array(data)
    x = data[:,:-1]
    y = data[:,-1]
    return np.array(x), np.array(y)

# Función para el preprocesado de los datos
def preprocessing_data(x):
    #Aumentamos dimensionalidad
    polynomical = PolynomialFeatures(3)
    x = polynomical.fit_transform(x)

    #Normalizamos los datos
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    
    return x

# Pintamos la nube de puntos de los datos
def plot_cloud_points(x,y,x_title,y_title,title):    
    x = np.array(x,np.float64)
    y = np.array(y,np.float64)
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()
    
# Función para mostrar gráfico de barras
def plot_bars_charts(bars,models,num_bars,title):
    bars = np.array(bars,np.float64)
    plt.bar(np.arange(num_bars),bars)
    plt.xticks(np.arange(num_bars),models,rotation=45)
    plt.title(title)
    plt.show()


# Muestra el mejor conjunto de parámetros con el que ha conseguido el porcentaje medio (valor medio adquirido tras el 5-fold cross-validation) 
# de mejor resultado
def draw_best_model(model):
    print("Best parameters: ", model.best_params_)
    print("Best cross-validation result: ", model.best_score_)
    print("Ein :", 1-model.best_score_)
    

#------------------------------ AJUSTE DE MODELOS ------------------------------

# Ajusta los datos mediante el modelo Ridge sobre los datos test
# con los mejores parámetros obtenidos
def adjust_Ridge(x_train,y_train,x_test,y_test,model):
    modelo = linear_model.Ridge(**model.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return predictions,score, modelo


# Ajusta los datos mediante el modelo Lasso sobre los datos test
# con los mejores parámetros obtenidos
def adjust_Lasso(x_train,y_train,x_test,y_test,model):
    modelo = linear_model.Lasso(**model.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return predictions, score, modelo


# Ajusta los datos mediante el modelo Regresión lineal sobre los datos test
# con los mejores parámetros obtenidos
def adjust_linearRegression(x_train,y_train,x_test,y_test,model):
    modelo = linear_model.LinearRegression(**model.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return predictions, score, modelo

    
#------------------------------ ESTUDIO DE MODELOS ------------------------------

# Establecemos los diferentes hyperparametros ha estudiar, generamos el modelo Ridge y mediante la funció de sklearn
# GridSearchCV realizamos las diferentes validaciones (5-fold cross-validation) con las distintas combinaciones de los hiperparámetros
# devolviendonos un modelo compuesto por todos los porcentajes medios de cada validación con sus respectivos parámetros y otra información
# util
def Ridge(x_train,y_train):
    parameters = [{'alpha':[0.1,0.05,0.01,0.005,0.001],'solver':['cholesky','sag'],'tol':[1e-3,1e-4]}]
    ridge = linear_model.Ridge(max_iter=100000)
    model = GridSearchCV(ridge,parameters,cv=5,scoring='r2')
    model.fit(x_train,y_train)
    draw_best_model(model)
    return model


# Establecemos los diferentes hyperparametros ha estudiar, generamos el modelo Lasso y mediante la funció de sklearn
# GridSearchCV realizamos las diferentes validaciones (5-fold cross-validation) con las distintas combinaciones de los hiperparámetros
# devolviendonos un modelo compuesto por todos los porcentajes medios de cada validación con sus respectivos parámetros y otra información
# util
def Lasso(x_train,y_train):
    parameters = [{'alpha':[0.1,0.05,0.01,0.005,0.001],'selection':['random','cyclic'],'tol':[1e-3,1e-4]}]
    lasso = linear_model.Lasso(max_iter=100000)
    model = GridSearchCV(lasso,parameters,cv=5,scoring='r2')
    model.fit(x_train,y_train)
    draw_best_model(model)
    return model


# Establecemos los diferentes hyperparametros ha estudiar, generamos el modelo de regresión lineal y mediante la funció de sklearn
# GridSearchCV realizamos las diferentes validaciones (5-fold cross-validation) con las distintas combinaciones de los hiperparámetros
# devolviendonos un modelo compuesto por todos los porcentajes medios de cada validación con sus respectivos parámetros y otra información
# util
def model_linearRegression(x_train, y_train):
    parameters = [{'fit_intercept':[True],'normalize':[True,False]}]    
    linear_regression = linear_model.LinearRegression()
    model = GridSearchCV(linear_regression, parameters, cv=5, iid=False)
    model.fit(x_train,y_train)
    draw_best_model(model)
    return model

        
#------------------------------ MAIN ------------------------------
    
# Cargamos los datos
datos = np.loadtxt('datos/airfoil_self_noise.dat')

# Separamos datos y etiquetas
col = len(datos[0])
x = datos[:,0:col-1]
y = datos[:, col-1]

# Generamos la división en datos de entrenamiento y prueba
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=1)


#Procesamos los datos
x_train_aux = preprocessing_data(x_train)
x_test_aux = preprocessing_data(x_test)


print("Gráfica de datos\n")
#Dibujamos la nube de puntos de los datos
plot_cloud_points(datos[:,0],datos[:,5],"Frequency","Sound pressure","Frequency vs Sound pressure")
plot_cloud_points(datos[:,1],datos[:,2],"Angle attack","Chord legnth","Angle attack vs Chord legnth")
plot_cloud_points(datos[:,3],datos[:,4],"Free-steam velocity","Suction side displacement thickness","Free-steam velocity vs Suction side displacement thickness")
input("\n--- Pulsar tecla para continuar ---\n")

# Creamos modelo de regresión lineal
print("\n## Linear Regression ##")
model_linearRegression = model_linearRegression(x_train_aux,y_train)

input("\n--- Pulsar tecla para continuar ---\n")

# Creamos modelo Ridge
print("\n## Ridge ##")
model_ridge = Ridge(x_train_aux,y_train)

input("\n--- Pulsar tecla para continuar ---\n")

# Creamos modelo Lasso
print("\n## Lasso ##")
model_lasso = Lasso(x_train_aux,y_train)


input("\n--- Pulsar tecla para continuar ---\n")
# Mostramos el gráfico de barras de los scores obtenidos por los modelos
bars = [model_linearRegression.best_score_,model_ridge.best_score_,model_lasso.best_score_]
models = ['LinearRegresssion','Ridge','Lasso']
plt.axis((0,1,0,1))
plot_bars_charts(bars,models,3,"Comparative score (cross-validation)")

#------------------------------ AJUSTE DEL MODELO FINAL ------------------------------

input("\n--- Pulsar tecla para continuar ---\n")
print("\n-------------------- FINAL ADJUST  MODEL --------------------")
# Finalmente elegimos el modelo final para ajustar, dicho modelo será el que haya obtenido mejores resultados 
# con sus hiperparametros correspondientes
if model_ridge.best_score_ > model_lasso.best_score_ and model_ridge.best_score_ > model_linearRegression.best_score_:
    print("\n\n**Final adjust: Ridge**")
    predictions, best_score_final, modelo = adjust_Ridge(x_train_aux, y_train, x_test_aux, y_test,model_ridge)
elif model_lasso.best_score_> model_ridge.best_score_ and model_lasso.best_score_ > model_linearRegression.best_score_: 
    print("\n\n**Final adjust: Lasso**")
    predictions, best_score_final, modelo = adjust_Lasso(x_train_aux, y_train, x_test_aux, y_test,model_lasso)
else:
    print("\n\n**Final adjust: Linear Regression**")
    predictions, best_score_final, modelo = adjust_linearRegression(x_train_aux, y_train, x_test_aux, y_test, model_linearRegression)


# Mostramos los resultados finales del modelo seleccionado
print("\n# Final Result #")
print("   -Hit percentage: ",best_score_final)
print("   -Eout: ", 1-best_score_final)  
print("\n")

input("\n--- Pulsar tecla para continuar ---\n")
# Mostramos las métricas calculadas
print("\n# METRICS #")      
#Recall no da información sobre el rendimiento de un clasifcador con respecto a los datos mal clasificados
print("   -MSE: ", metrics.mean_squared_error(y_test,predictions))
print("   -MAE: ", metrics.mean_absolute_error(y_test,predictions)) 
print("   -Coefficient of determination : ", metrics.r2_score(y_test,predictions))











