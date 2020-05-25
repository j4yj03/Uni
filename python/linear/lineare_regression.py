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

    mse = 0.5 * np.mean((y_true - y_pred) ** 2)
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
    #print("a=",alpha," -- ",X_ext.T.dot(X_ext) + alpha * IdentityMatrix)
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

    #Xs=(X[:,:]-mean[:])/std[:] if X.ndim > 1 else (X[:]-mean)/std
    Xs = (X-mean)/(std)
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
def LR_gradient_descent(X, y, theta0, nmax=10000, eta=0.01):
    assert eta > 0, 'eta kleiner 0'
    theta=theta0

    for i in range(nmax):
        #differenz zwischen y und h(theta) ermitteln
        loss = (Ridge_predict(X, theta) - y)
        # Vektor der Ableitungen (Gradientenvektor)
        gradient = 1/len(y) * extend_matrix(X).T.dot(loss)
        #theta um eta in richtung des gradientenabstieges anpassen
        theta = theta - eta * gradient
    #kosten
    J = np.mean(loss**2)

    return theta, J


def LR_gradient_descent_hist2(X, y, theta0, nmax=100000, eta=0.01):
    assert eta > 0, 'eta kleiner 0'
    theta=theta0

    thetas = []
    preds = []
    costs = []
    counter = 0
    for i in range(nmax):
        #differenz zwischen y und h(theta) ermitteln
        for j in range(len(y)):
            #differenz zwischen y und h(theta) ermitteln

            Xext = np.hstack([1,X[j]])
            #print("extended: ",Xext)
            loss = (theta.T.dot(Xext) - y[j])
            cost = np.mean(loss**2)
            #print("loss: ",loss)
            #gradient = 1/len(y) * np.gradient(min(cost))
            gradient= 1/len(y) * loss * Xext
            #print(counter, gradient)
            #print("gradient: ",gradient)
            # Vektor der Ableitungen (Gradientenvektor)
            #gradient = 1//len(y) * extend_matrix(X).T.dot(loss)
            #theta um eta in richtung des gradientenabstieges anpassen
            theta = theta - eta * gradient

            thetas.append(theta)
            #preds.append(y_pred)
            costs.append(cost)


            if len(thetas)> 10:
                if np.all(costs[-4:] == cost):
                    print('GD converged!')
                    break
        counter = i
        #print(counter)

    return thetas, costs#, preds, counter

def LR_gradient_descent_hist(X, y, theta0, nmax=100000, eta=0.0001):
    assert eta > 0, 'eta kleiner 0'
    theta=theta0

    thetas = []
    preds = []
    costs = []
    counter = 0
    for i in range(nmax):
        #differenz zwischen y und h(theta) ermitteln
        y_pred = Ridge_predict(X, theta)
        loss = (y_pred - y)
        cost = np.mean(loss**2)
        # Vektor der Ableitungen (Gradientenvektor)
        gradient = extend_matrix(X).T.dot(loss)
        #theta um eta in richtung des gradientenabstieges anpassen
        theta = theta - eta/len(y) * gradient

        thetas.append(theta)
        preds.append(y_pred)
        costs.append(cost)
        counter = i
        #print(len(thetas))
        if len(thetas)> 10:
            if np.all(costs[-4:] == cost):
                print('GD converged!')
                break

    return thetas, costs, preds, counter
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
