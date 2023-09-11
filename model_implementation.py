# -*- coding: utf-8 -*-

'''
Implementation of Classification Models
----------------------------------------------------------------------------------------------------------------

- Francia Garcia Romero

A logistic regression and random forest model are implementes using sklearn. The dataset has not changed, but,
instead of predicting whether a song belongs to a certain genre or not, we will work with a multiclass problem
comprising 4 different genres instead of 1.

The main problem in the previous implementation of logistic regression was the unbalance of classes causing underfitting.
Because of that, in the data_preprocessing.ipynb file, a new section  (Balance of Classes) to address this issue
was added. In that section, we eliminate 6 out of 10 classes, because they only represent 20% of the data together.

With the 4 remaining classes, because there are no longer "insignificant" classes, we apply oversampling and
undersampling (according to the specific needs of each class) to ensure a dataframe with 25% of each class.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('music_data.csv', index_col = 0)
# Target variable and features
X = df.drop('Class', axis=1)
y = df['Class']

# Train, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# -------------------------------------------------------------------------------------------


# Logistic Regression Implementation
def LogisticRegressionModel():
  model_lr = LogisticRegression(
      penalty='l2',
      C=1.0,
      solver='lbfgs',
      max_iter=50,
      random_state=42,
  )

  # Fitting of model
  model_lr.fit(X_train, y_train)

  print('='*70, '\nLogistic Regresion (before tuning)')
  print('='*70)

  # Evaluating with train data
  print('Evaluation with Training Data:')
  y_train_pred = model_lr.predict(X_train)
  evaluate(y_train, y_train_pred)

  # Evaluation with test data
  print('\nEvaluation with Testing Data:')
  y_test_pred = model_lr.predict(X_test)
  evaluate(y_test, y_test_pred)

  print('-------------------- Underfitting-------------------------\n')

  print('Enhancing model using Grid Search with Cross Validation...')

  # Define a grid of hyperparameters to search
  param_grid_lr = {
      'penalty': ['l1', 'l2'],
      'C': [0.001, 0.01, 0.1, 1, 10, 100],
      'solver': ['liblinear', 'saga'],
      'max_iter': [100, 500, 1000],
  }

  # Perform grid search with cross-validation
  grid_search_lr = GridSearchCV(estimator=model_lr, param_grid=param_grid_lr, cv=5, scoring='accuracy', verbose=0)
  grid_search_lr.fit(X_train, y_train)

  # Get the best hyperparameters
  best_params = grid_search_lr.best_params_

  # Train a Logistic Regression model with the best hyperparameters
  best_logistic_regression = LogisticRegression(**best_params)
  best_logistic_regression.fit(X_train, y_train)

  # Evaluating with calculated predictions
  print('\nEvaluation based on Predictions from Training Data:')
  y_train_pred = best_logistic_regression.predict(X_train)
  evaluate(y_train, y_train_pred)

  print('\nEvaluation based on Predictions from Testing Data:')
  y_test_pred = best_logistic_regression.predict(X_test)
  evaluate(y_test, y_test_pred)


# -------------------------------------------------------------------------------------------

# Random Forest Implementation
def RandomForestModel():
  # Fitting of model
  model = RandomForestClassifier(n_estimators=200, max_depth = None, random_state=42)
  model.fit(X_train, y_train)

  print('='*70, '\nRandom Forest (before tuning)')
  print('='*70)
  # Evaluating with train data
  print('Evaluation based on Predictions from Training Data:')
  y_train_pred = model.predict(X_train)
  evaluate(y_train, y_train_pred)

  # Evaluation with test data
  print('\nEvaluation based on Predictions from Testing Data:')
  y_test_pred = model.predict(X_test)
  evaluate(y_test, y_test_pred)


  print('-------------------- Overfitting-------------------------\n')

  print('Feature Selection was implemented, but best subset of features was the original set.\n')
  print('--------------> See model_implementation.ipynb for more information.\n')

  print('Cross Validation to find optimal hyperparameters was implemented.\n')
  print('--------------> See model_implementation.ipynb for complete process.\n')
  print('Best Hyper-parameters found: N-Estimators: 20, Max-Depth: 15')
  best_max_depth = 15
  best_n_estimators = 20

  # Cross Validation for adjusting and evaluating the model.
  rf_classifier = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
  print('\nEvaluation based on Predictions from Training Data')
  for metric in ['accuracy', 'precision_macro', 'recall_macro']:
    # Perform 5-fold cross-validation (you can change the number of folds with the cv parameter)
    scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring = metric)

    # Calculate and print the mean and standard deviation of the cross-validation scores
    print(metric, ': ', round(scores.mean() * 100, 2), '%')
  # Fitting of adjusted model
  rf_classifier.fit(X_train, y_train)

  print('\nEvalution based on Predictions from Testing Data:')
  # Evaluating with test data
  y_test_pred = rf_classifier.predict(X_test)
  evaluate(y_test, y_test_pred)


# ------------------------------------------------------------------------------------------
# Function to evaluate a model based on the expected results and the actual results.
def evaluate(y_true, y_pred):
  # Calculate accuracy
  accuracy = accuracy_score(y_true, y_pred)
  print(f'accuracy: {accuracy * 100:.2f}%')

  # Calculate precision for each class separately and then take the average
  precision = precision_score(y_true, y_pred, average='macro')
  print(f'precision: {precision * 100:.2f}%')

  # Calculate recall for each class separately and then take the average
  recall = recall_score(y_true, y_pred, average='macro')
  print(f'recall: {recall * 100:.2f}%')

  # Calculate the confusion matrix
  conf_matrix = confusion_matrix(y_true, y_pred)
  print("Confusion Matrix:")
  print(conf_matrix)


# -------------------------------------------------------------------------------------------



def main():
  LogisticRegressionModel()
  print('\n\n\n\n\n\n\n\n\n')
  RandomForestModel()

main()
