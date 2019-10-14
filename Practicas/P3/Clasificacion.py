# -*- coding: utf-8 -*-
"""
@author: SERGIO
"""

import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from scipy import interp
from sklearn import linear_model, metrics, preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

#------------------------------ FUNCIONES  ------------------------------

# Función de lectura de los datos: lee los datos del fichero y divide en datos y etiquetas
def readData(rute):	
    data = pd.read_csv(rute,header=-1)
    data = np.array(data)
    x = data[:,:-1]
    y = data[:,-1]
    return np.array(x), np.array(y)


# Dibuja algunos digitos del dataset de digitos manuscritos 
def draw_data(x_train):    
    plt.figure(figsize=(10,5))
    for i in range(10):    
        plot = plt.subplot(2,5,i+1)
        plot.imshow(np.split(x_train[i],8),cmap='cividis')
        plot.set_xticks(())
        plot.set_yticks(())    
    plt.show()


# Función para el preprocesado de los datos
def preprocessing_data(x):
    #Eliminamos los datos sin variabilidad
    unusuable = VarianceThreshold(0.1)
    x = unusuable.fit_transform(x)

    #Normalizamos los datos
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    return x


# Pinta las curvas de ROC de las diferentes clases del conjunto de datos
def roc_curve_multiclass(x,y,model):
    y = preprocessing.label_binarize(y,np.unique(y))
    num_clas = y.shape[1]
    fpr = {}
    tpr = {}
    score = model.decision_function(x)

    for i in range(num_clas):
        fpr[i], tpr[i],_ = metrics.roc_curve(y[:,i],score[:,i])
    fpr["Micro"],tpr["Micro"],_ = metrics.roc_curve(y.ravel(),score.ravel())
    fpr_a = np.unique(np.concatenate([fpr[i] for i in range(num_clas)]))
    mean_tpr= np.zeros_like(fpr_a)
    for i in range(num_clas):
        mean_tpr += interp(fpr_a, fpr[i], tpr[i])
    mean_tpr /= num_clas
    fpr["Macro"] = fpr_a
    tpr["Macro"] = mean_tpr
    plt.figure()
    colors = itertools.cycle(['aqua','darkorange','cornflowerblue', 'lime','crimson','lightpink','darkgreen','salmon','sienna','bisque'])
    for i, color in zip(range(num_clas),colors):
        plt.plot(fpr[i],tpr[i], color = color, lw=3, label = "ROC class {}".format(i))
    plt.plot([0,1],[0,1], color='navy',lw=2,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positives')
    plt.ylabel('True Positives')
    plt.title("ROC curve")
    plt.legend()
    plt.show()
    

# Dibuja la matriz de confusion
def draw_confusion_matrix(predictions,y_test):
    #creamos la matriz de confusión a partir de las etiquetas reales y las etiquetas de la predicción
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(confusion_matrix,cmap='Blues', annot=True, fmt=".3f",linewidths=.5,square=True)
    plt.xlabel('predict target')
    plt.ylabel('real target')
    plt.title('Confusion Matrix',size=15)
    plt.show()
    return confusion_matrix


#Pintamos la matriz de confusión resaltando en blanco las zonas donde
#se han predecido las etiquetas de cada dígito
def draw_confusion_matrix_contrast(confusion_matrix):
    plt.figure(figsize=(9,9))
    plt.matshow(confusion_matrix,cmap=plt.cm.gray)
    plt.title("Confusion matrix ")
    plt.xlabel('predict target')
    plt.ylabel('real target')
    plt.show()


# Pintamos la nube de puntos de las diferentes clases del dataset
def plot_cloud_points(x,y):
    plt.figure()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(x_train_aux)
    plt.scatter(proj[:,0],proj[:,1],c=y_train,cmap="Paired")
    plt.colorbar()
    plt.title("Cloud Points Multiclass")
    plt.show()
    
# Función que dibuja gráfico de barras
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

# Ajusta los datos, clasifica sobre los datos de prueba mediante el algoritmo
# de regresion logistica con los mejores parámetros obtenidos
def adjust_data_LR(x_train,y_train,x_test,y_test,model):
    modelo = linear_model.LogisticRegression(**model.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return predictions, score, modelo


# Ajusta los datos, clasifica sobre los datos de prueba mediante el algoritmo
# Perceptron con los mejores parametros obtenidos
def adjust_data_PLA(x_train,y_train,x_test,y_test,model):
    modelo = linear_model.Perceptron(**model.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return predictions, score, modelo


# Ajusta los datos, clasifica sobre los datos de prueba mediante el algoritmo
# de SVM con los mejores parámetros obtenidos
def adjust_SVM(x_train,y_train,x_test,y_test,model):
    modelo = svm.SVC(**model.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return predictions, score, modelo
    

#------------------------------ ESTUDIO DE MODELOS ------------------------------

# Establecemos los diferentes hiperparametros ha estudiar, generamos el modelo LR y mediante la funció de sklearn
# GridSearchCV realizamos las diferentes validaciones (5-fold cross-validation) con las distintas combinaciones de los hiperparámetros
# devolviendonos un modelo compuesto por todos los porcentajes medios de cada validación con sus respectivos parámetros y otra información
# util
def model_LR(x_train,y_train):
    parameters = [{'penalty':['l2'],'C':[1,10,100,1000,10000], 'tol':[1e-3,1e-4], 'solver':['newton-cg'], 'multi_class':['auto']}]
    logic_regression = linear_model.LogisticRegression(random_state=1,max_iter=1000)
    model = GridSearchCV(logic_regression, parameters, cv=5, scoring='accuracy')
    model.fit(x_train,y_train)
    draw_best_model(model)
    return model


# Establecemos los diferentes hiperparametros ha estudiar, generamos el modelo PLA-pocket y mediante la funció de sklearn
# GridSearchCV realizamos las diferentes validaciones (5-fold cross-validation) con las distintas combinaciones de los hiperparámetros
# devolviendonos un modelo compuesto por todos los porcentajes medios de cada validación con sus respectivos parámetros y otra información
# util
def model_PLA(x_train,y_train):
    parameters = [{'penalty':['l1','l2'],'alpha':[0.1,0.001,0.0001,0.00001],'tol':[1e-3,1e-4]}]
    perceptron = linear_model.Perceptron(random_state=1,max_iter=1000)
    model = GridSearchCV(perceptron, parameters, cv=5, iid=False, scoring='accuracy')
    model.fit(x_train,y_train)
    draw_best_model(model)
    return model


# Establecemos los diferentes hiperparametros ha estudiar, generamos el modelo SVM y mediante la funció de sklearn
# GridSearchCV realizamos las diferentes validaciones (5-fold cross-validation) con las distintas combinaciones de los hiperparámetros
# devolviendonos un modelo compuesto por todos los porcentajes medios de cada validación con sus respectivos parámetros y otra información
# util
def SVM(x_train, y_train):
    parameters = [{'gamma':['scale'],'C':[1,10,100,1000,10000],'kernel':['linear','poly'],'degree':[2,6],'tol':[1e-3,1e-4]}]
    svc = svm.SVC(max_iter=10000)
    model = GridSearchCV(svc, parameters, cv=5)
    model.fit(x_train,y_train)
    draw_best_model(model)
    return model

        
#------------------------------ MAIN ------------------------------
    
# Leemos los datos
x_train, y_train = readData('datos/optdigits.tra')
x_test, y_test = readData('datos/optdigits.tes')

# Convertimos a arraus de numpy
x_train = np.array(x_train, np.float64)
x_test = np.array(x_test, np.float64)

# Dibujamos algunos 
print("\n# DIGITS DATASET #")
draw_data(x_train)

#Procesamos los datos
x_train_aux = preprocessing_data(x_train)
x_test_aux = preprocessing_data(x_test)


input("\n--- Pulsar tecla para continuar ---\n")

#Dibujamos la nube de puntos de los datos procesados
plot_cloud_points(x_train_aux,y_train)

input("\n--- Pulsar tecla para continuar ---\n")

# Creamos modelo de regresión logistica
print("\n## Logistic Regression ##")
model_LR = model_LR(x_train_aux,y_train)

input("\n--- Pulsar tecla para continuar ---\n")

# Creamos modelo de Perceptron
print("\n## PLA ##")
model_PLA = model_PLA(x_train_aux,y_train)

input("\n--- Pulsar tecla para continuar ---\n")
# Pintamos la comparativa de los scores de los modelos lineales (gráfico de barras)
bars = [model_LR.best_score_, model_PLA.best_score_]
models = ['LR','PLA']
plot_bars_charts(bars,models,2,'Comparative score percentage')

#------------------------------ AJUSTE DEL MODELO FINAL ------------------------------

input("\n--- Pulsar tecla para continuar ---\n")
print("\n-------------------- FINAL ADJUST MODEL --------------------\n")
# Finalmente elegimos el modelo final para ajustar, dicho modelo será el que haya obtenido mejores resultados 
# con sus hiperparametros correspondientes
if model_LR.best_score_ > model_PLA.best_score_:
    print("\n\n** The best classifier is Logistic Regression **")
    predictions, best_score_final, modelo = adjust_data_LR(x_train_aux, y_train, x_test_aux, y_test,model_LR)
else:
    print("\n\n** The best classifier is PLA-Pocket **")
    predictions, best_score_final, modelo = adjust_data_PLA(x_train_aux, y_train, x_test_aux, y_test,model_PLA)


# Mostramos los valores finales del ajuste del Modelo
print("\n*FINAL RESULT*")
print("   -Hit percentage: ",best_score_final)
print("   -Eout: ", 1-best_score_final)  
print("\n")

# Mostramos la matriz de confusión
input("\n--- Pulsar tecla para continuar ---\n")
print("\nMETRICS")
print("# CONFUSION MATRIX #")
confusion_matrix = draw_confusion_matrix(predictions,y_test)

# Mostramos la matriz de confusión de contraste
input("\n--- Pulsar tecla para continuar ---\n")
print("# Confusion Matrix Contrast #")
draw_confusion_matrix_contrast(confusion_matrix)

# Mostramos las curvas de ROC de las diferentes clases
input("\n--- Pulsar tecla para continuar ---\n")
print("# ROC CURVE MULTICLASS #")
roc_curve_multiclass(x_test_aux,y_test,modelo)

# Mostramos las metricas
input("\n--- Pulsar tecla para continuar ---\n")
print("# MEAN RECALL AND PRECISION #")
#Recall no da información sobre el rendimiento de un clasifcador con respecto a los datos mal clasificados
print("   -Recall: ", np.mean((np.diag(confusion_matrix)/np.sum(confusion_matrix,axis=1))))
print("   -Precision: ",np.mean((np.diag(confusion_matrix)/np.sum(confusion_matrix,axis=0)))) 

input("\n--- Pulsar tecla para continuar ---\n")
# Dibujamos el gráfico de barras de la comparativa entre Recall y precisión
bars = [np.mean((np.diag(confusion_matrix)/np.sum(confusion_matrix,axis=1))),np.mean((np.diag(confusion_matrix)/np.sum(confusion_matrix,axis=0)))]
models = ['Recall','Precision']
plot_bars_charts(bars,models,2,"Comparative Recall vs Precision")

input("\n--- Pulsar tecla para continuar ---\n")

# Métricas obtenidas por las diferentes clases
print("# METRICS CLASS #")
print(metrics.classification_report(y_test,predictions))

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------ MODELO SELECCIONADO vs SVM ------------------------------

# Creamos modelo de SVM
print("\n## SVM ##")
model_SVM = SVM(x_train_aux,y_train)

input("\n--- Pulsar tecla para continuar ---\n")
print("# COMPARATIVE WITH SVM #")
# Ajuste del modelo SVM
predictionsSVM, best_score_SVM, modeloSVM = adjust_SVM(x_train_aux, y_train, x_test_aux, y_test, model_SVM)

# Mostramos los resultados del ajuste
print("\n-Hit percentage SVM (ADJUST): ", best_score_SVM)
print("-Hit percentage FinalLinearModel (ADJUST): ", best_score_final)

input("\n--- Pulsar tecla para continuar ---\n")
# Mostramos el gráfico de barras de la comparativa entre el score ajustado de SVM y
# el ajuste del mejor modelo lineal (LR o PLA)
bars = [best_score_final,best_score_SVM]
models = ['Best Lineal Model','SVM']
plot_bars_charts(bars,models,2,'Comparative score percentage (ADJUST)')

