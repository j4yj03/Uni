# -*- coding: utf-8 -*-

# gradient_descent
#
# Routinen zur multivariaten linearen Regression (Skalierung, Gradienten-
# Abstiegsverfahren) für die  Modellfunktion
#
#   h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
#
# und die Kostenfunktion
#
#   J(theta) = 1/(2m) sum_(i=1)^m ( h_theta(x^(i)) - y^(i) )^2
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
    X_ext = np.c_[np.ones((np.size(X,0),1)),X]
    return X_ext


#%% StandardScaler_fit

# Berechnet Mittelwert und Standardabweichung für Skalierung
#
# mean, std = StandardScaler_fit(X)
#
# Eingabe:
#   X       Matrix m x n (numpy.ndarray)
#
# Ausgabe
#   mean    Vektor der Länge n der spaltenweisen Mittelwerte (numpy.ndarray)
#           mean_j = 1/m sum_(i=1)^m x^(i)_j
#   std     Vektor der Länge n der spaltenweisen Standardabweichung (numpy.ndarray)
#           std_j = 1/m sum_(i=1)^m (y^(i)_j - mean_j)^2
#
# Hinweis: siehe entsprechende Routinen in numpy
#
def StandardScaler_fit(X):
    # TODO: Berechne mean, std
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    return mean, std


#%% StandardScaler_transform

# Berechnet Mittelwert und Standardabweichung für Skalierung
#
# Xs = StandardScaler_transform(X)
#
# Eingabe:
#   X       Matrix m x n (numpy.ndarray)
#   mean    Vektor der Länge n der spaltenweisen Mittelwerte (numpy.ndarray)
#   std     Vektor der Länge n der spaltenweisen Standardabweichung (numpy.ndarray)
#
# Ausgabe
#   Xs      Matrix m x n der spaltenweise skalierten Werte (numpy.ndarray)
#           Xs_(i,j) = (X_(i,j) - mean_j)/std_j
#
def StandardScaler_transform(X, mean, std):
    # TODO: Berechne Xs

    Xs=(X[:,:]-mean[:])/std[:] if X.ndim > 1 else (X[:]-mean)/std

    return Xs

def Ridge_predict(X, theta):

    y = extend_matrix(X).dot(theta)

    return y

#%% LR_gradient_descent

# Gradientenabstiegsverfahren für die lineare Regression
#
# theta, J = LR_gradient_descent(X, y, theta0, nmax, eta)
#
# Eingabe:
#   X       Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y       Vektor der Länge m der Zielwerte (numpy.ndarray)
#   theta0  Startvektor der  Länge n+1 der optimalen Parameter (numpy.ndarray)
#   nmax    Anzahl der Iterationen (int)
#   eta     Schrittweite eta>0 (float)
#
# Ausgabe
#   theta   Aktueller Vektor der Länge n+1 der optimalen Parameter (numpy.ndarray)
#   J       Aktueller Wert der Kostenfunktion (float)
#
def LR_gradient_descent(X, y, theta0, nmax=1000000, eta=0.0005):
    assert eta > 0, 'eta kleiner 0'
    theta=theta0

    for i in range(nmax):
        #differenz zwischen y und h(theta) ermitteln
        loss = (Ridge_predict(X, theta) - y)
        # Vektor der Ableitungen (Gradientenvektor)
        gradient = extend_matrix(X).T.dot(loss)
        #theta um eta in richtung des gradientenabstieges anpassen
        theta = theta - eta/len(y) * gradient
    #kosten
    J = np.mean(loss**2)

    return theta, J
#
# def LR_gradient_descent2(X, y, theta0, nmax=1000000, eta=0.0001):
#     # TODO: berechne theta, J
#
#     theta=theta0
#
#     for i in range(nmax):
#         # kosten ermitteln
#         dif = Ridge_predict(X, theta)-y
#         #print(dif)
#         cost = np.power((Ridge_predict(X, theta)-y),2)/2
#         #print(len(cost))
#         # theta
#         theta = theta - eta/len(y) * np.sum(cost)
#
#
#     J = cost
#     return theta, J
