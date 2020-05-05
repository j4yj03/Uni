# -*- coding: utf-8 -*-

# lineare_regression
#
# Routinen zur Berechnung der multivariaten linearen Regression mit Modell-
# funktion
#
#   h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
#
# und Kostenfunktion
#
#   J(theta) = 1/(2m) sum_i=1^m ( h_theta(x^(i)) - y^(i) )^2
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

#%% extend_matrix

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

    # Anzahl der Zeilen bestimmt die groesse des Stack
    #stack = np.array([[1] for x in range(np.size(X,0))])
    # Original wird and den Stack geklebt
    #X_ext = np.hstack([stack,X])

    X_ext = np.c_[np.ones((np.size(X,0),1)),X]

    return X_ext

#%% train_test_split (wird bereit gestellt)

# Teilt den Datensatz in Training (Anteil frac) und Test (Rest)
#
# [Xtrain, Xtest, ytrain, ytest] = train_test_split(X)
#
# Eingabe:
#   X      Matrix m x n (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   frac   Anteil im Trainingsset 0 <= frac <= 1
#
# Ausgabe
#   Xtrain Featurematrix Trainingsset
#          mtrain x n mit mtrain = frac * m (numpy.ndarray)
#   Xtest  Featurematrix Testset
#          mtest = m - mtrain (numpy.ndarray)
#   ytrain Vektor Zielwerte Trainingsset Länge mtrain (numpy.ndarray)
#   ytest  Vektor Zielwerte Testset Länge mtest (numpy.ndarray)
#
def train_test_split(X, y, frac, seed=42):
    m = X.shape[0]
    np.random.seed(seed)
    index = np.arange(m)
    np.random.shuffle(index)
    cut = int(m*frac)

    return X[index[:cut],:], X[index[cut:],:], y[index[:cut]], y[index[cut:]]

#%% QuadraticFeatures_fit_transform

# Fügt der Featurematrix quadratische und gemischte Features hinzu
#
# Xq = QuadraticFeatures_fit_transform(X)
#
#        [ |        |    |        |            |       |       |          |   ]
#   Xq = [x_1, ... x_n, x_1^2, x_1*x_2, ... x_1*x_n, x_2^2, x_2*x_3, ... x_n^2]
#        [ |        |    |        |            |       |       |          |   ]
#
# Eingabe:
#   X      Featurematrix m x n (numpy.ndarray)
#
# Ausgabe
#   Xq     Featurematrix (m+m*(m+1)/2) x n (numpy.ndarray)
#
def QuadraticFeatures_fit_transform(X):
    # TODO: berechne Xq
    return Xq


#%% mean_squared_error

# Berechnung des mittleren Fehlerquadrats
#
# mse = mean_squared_error(y_true, y_pred)
#
# Eingabe:
#   y_true  Vektor der Länge m der wahren Zielwerte (numpy.ndarray)
#   y_pred  Vektor der Länge m der vorhergesagten Zielwerte (numpy.ndarray)
#
# Ausgabe
#   mse    Mittleres Fehlerquadrat mse = 1/m sum_(i=1)^m (y_true_i-y_pred_i)^2
#
def mean_squared_error(y_true, y_pred):
    # TODO: berechne mse
    mse =  np.mean((y_true - y_pred) ** 2)
    return mse


#%% Ridge_fit

# Berechnung der optimalen Parameter der multivariaten regularisierten linearen
# Regression mithilfe der Normalengleichung.
#
# theta = Ridge_fit(X, y, alpha)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   alpha  Regularisierungsparameter (Skalar)
#
# Ausgabe
#   theta  Vektor der  Länge n+1 der optimalen Parameter (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix und np.linalg.solve zur Lösung des
#   linearen Gleichungssystems
#
def Ridge_fit(X, y, alpha):
    # TODO: berechne theta
    return theta


#%% Ridge_predict

# Berechnung der Vorhersage der der multivariaten regularisierten linearen
# Regression.
#
# y = Ridge_predict(X,theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   y      Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix.
#
def Ridge_predict(X, theta):
    # TODO: berechne y
    return y


#%% LR_fit

# Berechnung der optimalen Parameter der multivariaten linearen Regression
# mithilfe der Normalengleichung.
#
# X_ext = LR_fit(X, y)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#
# Ausgabe
#   theta  Vektor der  Länge n+1 der optimalen Parameter (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix und np.linalg.solve zur Lösung des
#   linearen Gleichungssystems
#
def LR_fit(X, y):

    X_ext = extend_matrix(X)

    theta = np.linalg.solve(X_ext.T.dot(X_ext),X_ext.T.dot(y))
    #theta = np.linalg.inv(X_ext.T.dot(X_ext)).dot(X_ext.T).dot(y) #ueber Inverse nicht optimal

    return theta


#%% LR_predict

# Berechnung der Vorhersage der der multivariaten linearen Regression.
#
# y = LR_predict(X,theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   y      Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix.
#
def LR_predict(X, theta):

    y = extend_matrix(X).dot(theta)

    return y


#%% r2_score

# Berechnung des Bestimmtheitsmaßes R2
#
# y = r2_score(X, y, theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   r2     Bestimmtheitsmaß R2 (Skalar)
#
# Hinweis: Benutzen Sie LR_predict
#
def r2_score(X, y, theta):

    y_pred = LR_predict(X, theta)

    sqr = np.sum((y - y_pred) ** 2)
    sqt = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (sqr/sqt) if sqt != 0 and sqr!=0 else 0

    return r2
