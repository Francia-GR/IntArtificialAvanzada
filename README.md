# Inteligencia Artificial Avanzada
Implementación de Modelos de Machine Learning

### **Files**:
  
* **regresion_logistica.py**: Contiene la implementación de un modelo de clasificación por regresión logística en Python. Utiliza el archivo *music_data.csv*.

* **model_implementation.py**: Contiene la implementación de un modelo de clasificación logística multiclase y un modelo de *random forest* utilizando la librería *sklearn* en Python. Utiliza el archivo *music_data.csv*.

* **model_implementation_explained.ipynb**: Contiene una descripción más detallada de la implementación de los modelos, y el proceso completo de cómo se encontró la configuración más apropiada para cada uno (selección de variables, grid search, cross-validation, gráficas del error del modelo en función de la complejidad del mismo). Utiliza el archivo *music_data.csv*.

* **ModelEvaluationReport**: Reporte final de la evaluación de los modelos. Análisis del sesgo y la varianza de los resultados, y descripción del manejo de *underfitting* u *overfitting** según correspondiera.
  
-----------------------------------------

* **data_preprocessing.ipynb**: Notebook que presenta el análisis y la limpieza inicial realizado sobre *train.csv* para obtener el archivo *music_data.csv*.

* **train.csv**: Datos originales tomados de [Music Genre Classification Challenge](https://www.kaggle.com/datasets/purumalgi/music-genre-classification). Solamente se utilizó el set de entrenamiento puesto que, al ser un reto de Kaggle, el set de *test* no tiene los valores reales de las observaciones a predecir.

* **music_data.csv**: Datos modificados a partir de los datos originales obtenidos de [Music Genre Classification Challenge](https://www.kaggle.com/datasets/purumalgi/music-genre-classification). Se hizo un preprocesamiento sencillo del archivo, en donde se omitieron columnas incompletas, o con poca significancia estadistica para la prediccion de la variable objetivo, los registros con valores faltantes (alrededor del 10%) se eliminaron y se estandarizaron los valores (min-max scaling). Asimismo, se realizó una descripción estadistica de las variables para la selección de el subset más apropiado para la implementación de modelos.

-----------------------------------------

* **Analisis_Normatividad.pdf**: Reporte acerca de la normatividad a la que está sujeto el conjunto de datos seleccionado para la implementación de regresión logística como solución para un problema de clasificación binaria.
