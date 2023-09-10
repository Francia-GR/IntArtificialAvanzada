'''
Modulo 2: Implementación de Técnica de Machine Learning
Francia Garcia Romero
A01769680

Logistic Regression Implementation
-------------------------------------------------------------------------------------------------------
(Reporte al final del documento)
'''
import pandas as pd
import numpy as np
import random
from math import e


def convergence(arr, difference = 0.000001):
  '''
  Args:
      arr (list[int]): Loss history
      difference (int, optional): Expected difference between last loss entries to consider it converges.

  Returns:
      (bool): Whether the values of loss are converging towards a certain value.
  '''
  if len(arr) < 2:
    return False
  if abs(arr[-2] - arr[-1]) < difference:
    return True
  return False



def sigmoid(x):
  '''
  Args:
      x (int, list[int]): Input value(s).

  Return:
      (float): Sigmoid function value as a function of x.
  '''
  return 1/(1 + e**(-x))



def train(X, y, learning_rate = 0.001, epochs = 10000, converg_limit = 0.00001): # init_weights
  '''
  This function adjusts weights for a Logistic Regression classifier using
  Gradient Descent.

  Args:
      X (np.ndarray): Matrix of M features for N observations (N, M).
      y (np.ndarray): Vector of actual values of N observations (N, ).
      learning_rate (float, optional): Step size for change by iteration for each weight
                                       parameter. Defaults to 0.001
      epochs (int, optional): Number of iterations to adjust weights. Defaults to 10, 000.
      converg_limit (float): Limit of difference between losses to stablish convergence.

    Returns:
      weights (np.ndarray): Adjusted weights.
      loss_history: Loss record throughout the training.

  '''
  # To take into account the bias, we will add a dummy variable to the features matrix X (-1 is placeholder)
  bias = np.ones(X.shape[0]).reshape(-1, 1)
  X = np.concatenate((X, bias), axis = 1)

  weights = np.ones(X.shape[1]) # Initialization of weights as zeros

  loss_history = []

  for epoch in range(epochs):

    # If the difference between the last registered losses is smaller than a certain number
    # the algorithm has reached convergence before epoch, so we stop it.

    if convergence(loss_history, converg_limit):
      # Print last 5 values to visualize convergence
      print("...")
      last_epoch = len(loss_history)

      for i in range(5, 0, -1):
        print('epoch: {} | loss: {}'.format(last_epoch - i, round(loss_history[last_epoch - i], 10)))
      print("Convergence reached.")
      break

    N = len(y)
    y_pred = sigmoid(X.dot(weights)) # Array of predictions: (N, ) where N -> number of observations

    # Gradient of Loss: Rates of change for Loss with respect to each parameter in 'weights' (dL/dw_i)
    # For Binary Cross-Entropy Loss: (1/N) * X.T * (y_pred - y)

    gradient = (1 / N) * (X.T.dot(y_pred - y))  # int * (M, N) * (N, ) -> (M, )

    # Gradient Descent: We update the weigths based on how does the error changes with respect to
    #                   each of them. If error increases as weight increases, we decrease the weight
    #                   and viceversa.
    weights = weights - learning_rate * gradient # (M, ) - int * (M, ) -> (M, )

    # Compute loss to check if it starts to converge towards a value, and store it.
    # For Binary Cross-Entropy Loss:

    loss = -(1/N) * np.sum(y.dot(np.log(y_pred)) + (1 - y_pred).dot(np.log(1 - y_pred)))
    loss_history.append(loss)

    # Showing progress of training
    if epoch % 10000 == 0:
        print('epoch: {} | loss: {}'.format(epoch, round(loss, 10)))

  return (weights, loss_history)



def predict_probability(features, weights):
  '''
  Calculates the probability an observation has of belonging to a class.

  Args:
      features (ndarray): Values of M features of N observations to predict. (N, M)
                          Both dimensions should be >= 1. For predictions of single
                          observations use features.reshape(1, -1) or [[values]].
      weights (ndarray):  Adjusted weights to the data. (M, )

  Returns:
      (ndarray[float]): Probabilities of observation of belonging to a class, calculated with a
               sigmoid soft-threshold.
  '''
  # Dummy variable for bias
  bias = np.ones(features.shape[0]).reshape(-1, 1)
  features = np.concatenate((features, bias), axis = 1)

  return sigmoid(features.dot(weights))



def predict(features, weights, threshold = 0.5, className = ''):
  '''
  Uses the probability computes by predict_probability to decide whether an observation
  belongs to a class based ona threshold.

  Args:
      features (ndarray): Values of M features of N observations to predict. (N, M)
                          Both dimensions should be >= 1. For predictions of single
                          observations use features.reshape(1, -1) or [[values]].
      weights (ndarray): Adjusted weights to the data. (M, )
      threshold (int, optional): Minimum probaility to consider an observation as part of a class.
                       Defaults to 0.5.
      className (str): Name of the class to be predicted

  Returns:
      (ndarray[int/str]): Whether an observation belong (1) or not (0) within a class. If className
                      is provided, it returns an array of strings with the name of the class.
  '''
  probabilities = predict_probability(features, weights)
  belongs_to_class = probabilities >= threshold

  # If className is specified, instead of 1 and 0, we will return 'className' and 'Not className'
  if className != '':
    return np.where(belongs_to_class, className, 'Not {}'.format(className))

  return belongs_to_class.astype(int)



def train_test_split(X, y, test_size = 0.3):
  '''
  Divides X and y into train and test set, choosing a random sample of the data.

  Args:
      X (np.ndarray): Matrix of M features for N observations (N, M).
      y (np.ndarray): Vector of actual values of N observations (N, ).
      test_size (float): Proportion of dataset that should be used for testing.
                         Defaults to 0.3.

  Returns:
      X_train, X_test, y_train, y_test (np.ndarray): Randomly divided dataset.
  '''
  N = len(y)
  sample_size = int(N * test_size)

  # Random sample (of specified size) of X and y for testing.
  random.seed(6)
  test_index = random.sample(range(N), sample_size)

  X_test = X[test_index]
  y_test = y[test_index]

  # The indices used for test_index are excluded to get the train sets.
  train_index = np.ones(N, dtype = bool) # [True, True, ... , True]
  train_index[test_index] = False

  X_train = X[train_index]
  y_train = y[train_index]

  return X_train, X_test, y_train, y_test



def evaluate(y, y_pred):
  '''
  Computes different metrics to evaluate the performance of the trained model
  based on the predicted values obtained.

  Args:
      y (np.ndarray): Actual values of sample used to predict.
      y_pred (np.ndarray): Values predicted by the trained model.

  Returns:
    metrics (dictionary): Dictionary containg confusion matrix, accuracy,
                          precision and recall of the result of the model.

  '''
  # Sum of real and predicted values. If sum == 2, it is a true positive,
  # if sum == 0, true negative, if sum == 1, the prediction was incorrect.
  true_positive_negative =  y + y_pred
  TP = np.count_nonzero(true_positive_negative == 2) # True Positive
  TN = np.count_nonzero(true_positive_negative == 0) # True Negative

  # Difference of real and predicted values. If diff == -1, it is a false positive,
  # if diff == 1, false negative, otherwise the prediction was correct.
  false_positive_negative = y - y_pred
  FP = np.count_nonzero(false_positive_negative == -1) # False Positive
  FN = np.count_nonzero(false_positive_negative == 1) # False Negative

  # Calculation of metrics
  confusion_matrix = np.array([[TP, FP],
                               [FN, TN]])
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  accuracy = (TP + TN) / (TP + FP + TN + FN)

  metrics = {'confusion matrix':confusion_matrix, 'precision':precision, 'recall': recall, 'accuracy':accuracy}

  return metrics



def main():
  df = pd.read_csv('music_data.csv', index_col = 0)

  # Separating features
  X = df[df.columns[0:-1]]
  X = np.array(X)

  # y will be whether a song is Pop (9)
  y = df['Class']
  y = y.apply(lambda x: 1 if (x == 9) else 0)
  y = np.array(y)

  # Spliting into train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

  # Training model
  weights, loss_list = train(X_train, y_train, learning_rate = 0.0001, epochs = 20000, converg_limit = 0)

  # Predictions for test data
  y_test_pred = predict(X_test, weights, threshold = 0.5)

  # Evaluation of Model
  metrics = evaluate(y_test, y_test_pred)

main()


'''
Reporte
------------------------------------------------------------------------------------------------------------

El modelo que se implentó fue el de regresión logistica para clasificación binaria. El dataset que se utilizó
tiene características que describen canciones que pertenecen a distintos géneros de música. Aunque el dataset
originalmente cuenta con 10 géneros, se opto por convertir este problema en uno de clasificacion binaria esta-
bleciendo 2 categorias más generales: canciones de rock y canciones que no son de rock. A partir de este nuevo
dataset transformado, se realizó una limpieza de los datos y una primera 'preparación' de los mismo. 

Respecto a la implementación del algoritmo, se siguio toda la lógica y matemática vista en clase con respecto
al modelo. Como se sabe, es una algoritmo iterativo que ajusta los parametros de la funcion que clasifica to-
mando en cuenta el error que se tiene al usar dichos parametros. Mas especificamente, se calcula la tasa de 
cambio del error con respecto a cada uno de los parametros, y se busca modificar el parametro de tal manera que
el error tienda a disminuir:

                            w_i <- w_i - alpha * (gradiente de la perdida)

donde alpha en la tasa de aprendizaje (step size).

La idea detras del algoritmo de gradiente descendiente es bastante intuitiva: si la tasa de cambio (derivada)
del error respecto a un parametro es positiva, quiere decir que si aumenta el parametro, aumenta el error, por
lo que debemos reducir (restar alpha * gradiente) el valor del parametro para reducir el error. Por el contrario, 
si la tasa de cambio del error respecto al parametro es negativa, quiere decir que entre más aumentemos el valor
del parámetro, el error disminuye, por lo que debemos aumentar(- alpha * (- derivada) = + alpha * derivada) el 
parametro en cuestion.

Habiendo comprendido esto, fue bastante sencillo manejar la estructura general del algoritmo. En cuando a la
funcion que se utilizó para el cálculo de la pérdida, se eligio la de Binary Cross-Entropy. No obstante, se 
pudo notar un comportamiento curioso en ciertas ocasiones. La funcion de perdida aumentaba y despues comenzaba a 
disminuir. Es muy probable que esto se deba a una mala interpretacion de la formula, o un error en la derivacion
manual para calcular el gradiente, por lo que para la siguiente entrega se espera corregir esta situación.


Conclusiones
------------------------------------------------------------------------------------------------------------

Respecto a los resultados obtenidos con la prediccion, se obtuvieron las siguientes metricas de evaluacion:

                                    {'confusion matrix': array([[   3,    3],
                                                                [1265, 3412]]),
                                    'precision': 0.5,
                                    'recall': 0.002365930599369085,
                                    'accuracy': 0.7292333973948324}

Como se pudo observar, en general no se obtuvo un buen desempeño al implementar este modelos para este conjunto
de datos en especifico. Esto fue debido principalmente a que, al ser un dataset con 10 categorias originalmente,
hay un gran desbalance en la cantidad de canciones que si pertenecen al genero Rock, y a las que no. Es por esto
que aunque tenemos un accuracy del 70%, en nuestra matriz de confusion podemos observar que hay un gran numero de
FALSOS NEGATIVOS: al haber demasiados casos en donde la cancion no pertenece a la clase, el modelo aprendió que
clasificando la mayoria como que no pertenece, va a acertar con más predicciones. El recall es muy bajo por lo
mismo: no tiene una gran cantidad de aciertos a casos positivos por que no hay tantos casos positivos. 

Otra razon que explica el pobre desempeño del modelo es la cantidad de variables. Inicialmente se tenian 25 
variables que fueron reducidas a 11, pero este sigue siendo un numero muy alto de features. Por ello, debe modificarse
el dataset para evitar este desbalance, o bien, hacer las modificaciones necesarias para realizar una regresión 
logística de múltiples clases, y aplicar técnicas de feature engineering.




'''
