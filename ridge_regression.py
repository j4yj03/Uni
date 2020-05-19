# -*- coding: utf-8 -*-

# ridge_fit
#
# Routinen zur Berechnung der regularisierten multivariaten linearen Regression
# mit polynomialen Features und Modellfunktion
#
#   h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
#
# und Kostenfunktion
#
#   J(theta) = 1/(2m) sum_(i=1)^m ( h_theta(x^(i)) - y^(i) )^2
#              + alpha/(2m) sum_(k=1)^n theta_k^2
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
def train_test_split(X, y, frac, seed):

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
#   degree Polynomengrad (default = 2)
#
# Ausgabe
#   Xq     Featurematrix (m+m*(m+1)/2) x n (numpy.ndarray)
#
from itertools import chain, combinations_with_replacement

def QuadraticFeatures_fit_transform(X, degree = 2):
    if isinstance(X, np.ndarray):
        length = np.size(X,0)
        X = X.T
    else:
        length = len(X[0])
    #Gemischte Feature Kombinationsmöglichkeiten für jeden Polynomengrad
    combinations = list(chain.from_iterable(combinations_with_replacement(X,i) for i in range(1, degree+1)))
    #leange der Liste entspricht ((m + degree)! / degree! + m! ) - 1
    Xq=np.empty([length, len(combinations)])

    for ind, vector in enumerate(combinations):
        #Produkt der Kombinationsmöglichkeiten
        Xq[:,ind] = np.prod(vector, axis=0, dtype=np.double)

    return Xq

# mean_squared_error

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

    mse = np.mean((y_true - y_pred) ** 2)
    #mse2 = 1/(len(y_true))*np.sum((y_true - y_pred) ** 2)
    #print(mse,mse2)
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


    X_ext = extend_matrix(X)

    IdentityMatrix = np.identity(X_ext.shape[1])
    IdentityMatrix[0][0] = 0
    print("a=",alpha," -- ",X_ext.T.dot(X_ext) + alpha * IdentityMatrix)
    theta = np.linalg.solve(X_ext.T.dot(X_ext) + alpha * IdentityMatrix, X_ext.T.dot(y))

    #print("X shape: {} X_ext shape: {} Identity shape: {} theta shape: {}".format(X.shape,X_ext.shape,IdentityMatrix.shape,theta.shape))

    return theta

def Ridge_fit2(X, y, alpha):
    #assert alpha > 0, "Error: file:" + str(file) + " line: 144"
    X = extend_matrix(X)
    M, N = X.shape
    regular_mx = np.zeros(shape = (N, N))
    np.fill_diagonal(a = regular_mx[1:,1:], val = 1)
    xtrans_x = X.T @ X
    xtrans_y = X.T @ y
    brackets = xtrans_x + alpha * (regular_mx)
    theta = np.linalg.solve(brackets, xtrans_y);
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

    y = extend_matrix(X).dot(theta)

    return y



def r2_score(y, y_pred):

    #y_pred = LR_predict(X, theta)

    sqr = np.sum((y - y_pred) ** 2)
    sqt = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (sqr/sqt) if sqt != 0 and sqr!=0 else 0

    return r2
