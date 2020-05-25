# -*- coding: utf-8 -*-

# logistische_regression
#
# Routinen zur Berechnung der multivariaten logistischen Regression
# mit Modellfunktion
#
#   h_theta(x) = sigma(theta_0 + theta_1 * x_1 + ... + theta_n * x_n)
#
# mit
#
#   sigma(t) = 1/(1+exp(-t))
#
# und Kostenfunktion
# 
#   J(theta) = -1/m sum_(i=1)^m (y^(i) log(h_theta(x^(i))) 
#                               + (1-y^(i)) log(1 - h_theta(x^(i))) 
#
# Der Vektor theta wird als
#
#   (theta_0, theta_1, ... , theta_n)
#
# gespeichert. Die Feature-Matrix mit m Daten und n Features als
#
#       [ - x^(1) - ]
#   X = [    .      ]    (m Zeilen und n Spalten)
#       [ - x^(m) - ]
#

import numpy as np

#%% extend_matrix (vom letzten Mal verwenden, wird nicht geprüft)

# Erweitert eine Matrix um eine erste Spalte mit Einsen
#
# X_ext = extend_matrix(X)
#
# Eingabe:
#   X      Matrix m x n (numpy.ndarray)
#
# Ausgabe
#   X_ext  Matrix m x (n+1) der Form [1 X] (numpy.ndarray)
#
def extend_matrix(X):
    # TODO: berechne X_ext
    return X_ext



#%% logistic_cost_function

# Berechnung der Kostenfunktion der logistischen Regression und deren 
# Gradienten
#
# J, Jgrad = logistic_cost_function(X,y, theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   J      Wert der regularisierten Kostenfunktion (Skalar)
#   Jgrad  Gradient der regularisierten Kostenfunktion (numpy.ndarray)
#
def logistic_cost_function(X,y, theta):
    # TODO: berechne J und Jgrad
    return J, Jgrad        


#%% LogisticRegression_fit

# Berechnung der optimalen Parameter der multivariaten logistischen Regression 
# mithilfe des Gradientenabstiegsverfahrens
#
# theta, J = LogisticRegression_fit(X,y,eta,tol)
#
# Die Iteration soll abgebrochen werden, falls 
#
#   || grad J || < tol
#
# gilt, wobei ||.|| die (euklidsche) Länge eines Vektors ist. Die Iteration
# soll abbrechen (mittels raise), falls die Kostenfunktion nicht fällt. Als
# Startvektor soll der Nullvektor gewählt werden.
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   eta    Learning rate (Skalar)
#   tol    Toleranz der Abbruchbedingung
#
# Ausgabe
#   theta   Aktueller Vektor der Länge n+1 der optimalen Parameter (numpy.ndarray)
#   J       Aktueller Wert der Kostenfunktion (float)
#
def LogisticRegression_fit(X,y,eta,tol):
    # TODO: berechne theta und J
    return theta, J


#%% LogisticRegression_predict

# Berechnung der Vorhersage der multivariaten logistischen Regression
#
# y = LogisticRegression_predict(X,theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   y      Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
#
def LogisticRegression_predict(X, theta):
    # TODO: berechne y
    return y
    

#%% accuracy_score

# Berechnung der Genauigkeit
#
# acc = accuracy_score(y_true,y_pred)
#
# Eingabe:
#   y_true Vektor der Länge m der wahren Werte (numpy.ndarray)
#   y_pred Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Ausgabe
#   acc    Genauigkeit (Skalar)
#
def accuracy_score(y_true,y_pred):
    # TODO: berechne acc
    return acc


#%% precision_score

# Berechnung der Präzision bzgl. der Klasse 1
#
# prec = precision_score(y_true,y_pred)
#
# Eingabe:
#   y_true Vektor der Länge m der wahren Werte (numpy.ndarray)
#   y_pred Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Ausgabe
#   prec    Genauigkeit (Skalar)
#
def precision_score(y_true,y_pred):
    # TODO: berechne prec
    return prec

#%% recall_score

# Berechnung des Recalls bzgl. der Klasse 1
#
# recall = recall_score(y_true,y_pred)
#
# Eingabe:
#   y_true Vektor der Länge m der wahren Werte (numpy.ndarray)
#   y_pred Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Ausgabe
#   recall Recall (Skalar)
#
def recall_score(y_true,y_pred):
    # TODO: berechne recall
    return recall# -*- coding: utf-8 -*-
