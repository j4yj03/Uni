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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
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
    X_ext = np.c_[np.ones((np.size(X,0),1)),X]
    return X_ext

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
    #berechne Wahrscheinlichkeit


    h = sigmoid(extend_matrix(X).dot(theta))
    y = (h >= 0.5).astype(int)

    return y
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
    #epsilon = 1e-10
    #print(theta)
    #y_pred = LogisticRegression_predict(X, theta)
    h = sigmoid(extend_matrix(X).dot(theta))

    J = - np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    Jgrad = 1/len(y) * extend_matrix(X).T.dot(h - y)

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
def LogisticRegression_fit(X,y, eta, tol):
    # TODO: berechne theta und J
    assert eta > 0, 'eta kleiner 0'

    X_ext = extend_matrix(X)

    theta = np.zeros(X_ext.shape[1])
    thetas = []
    costs = []
    J, Jgrad = logistic_cost_function(X,y, theta)

    while(np.linalg.norm(Jgrad) >= tol):

        #theta um eta in richtung des gradientenabstieges anpassen
        theta = theta - eta * Jgrad

        J, Jgrad = logistic_cost_function(X,y, theta)

        thetas.append(theta)
        costs.append(J)

        if len(thetas) > 2:
             if np.all(costs[-2:] == J):
                 raise "J fällt nich"

    return theta, J





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
def accuracy_score(y_true, y_pred):
    # TODO: berechne acc

    tn = np.sum((y_true == y_pred) & (y_true == 0))
    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fp = np.sum((y_true != y_pred) & (y_true == 0))
    fn = np.sum((y_true != y_pred) & (y_true == 1))

    acc = (tp+tn) / (tp+tn+fp+fn)

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
    # tp / tp + fp
    # TODO: berechne prec

    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fp = np.sum((y_true != y_pred) & (y_true == 0))

    prec = tp / (tp + fp)

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
    # tp / tp + fn
    # TODO: berechne recall
    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fn = np.sum((y_true != y_pred) & (y_pred == 0))

    recall = tp / (tp + fn)

    return recall# -*- coding: utf-8 -*-
