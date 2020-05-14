import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import lineare_regression as lreg
import gradient_descent as gd
import ridge_regression as ridge

if __name__ == "__main__":

    #univariat
    uni = pd.read_csv("./data/univariat.csv", sep=',')
    x_u = uni['x'].to_numpy()
    y_u = uni['y'].to_numpy()

    #multivariat
    multi = pd.read_csv("./data/multivariat.csv", sep=',')
    # x1_m = multi['x1'].to_numpy()
    # x2_m = multi['x2'].to_numpy()
    X_m = multi.iloc[:,0:2].to_numpy()
    y_m = multi['y'].to_numpy()

    #print(f'{X_m} {y_m}')

    #print(X_m.shape,y_m.shape)
    #print(x2,x2.shape)

    mean, std = gd.StandardScaler_fit(y_m)
    ys = gd.StandardScaler_transform(y_m, mean, std)

    print(f'ym: mean={mean} std={std}')

    X_m_2 = ridge.QuadraticFeatures_fit_transform(X_m, 2)
    mean, std = gd.StandardScaler_fit(X_m_2)
    Xs2 = gd.StandardScaler_transform(X_m_2, mean, std)
    print(f'Xm: mean={mean} std={std}')

    #print(f'{Xs}')
    theta_ridge = ridge.Ridge_fit(Xs2 ,ys, 0)

    print(f'\ntheta_normal:{theta_ridge}\n')

    theta0 = np.zeros(6)
    theta , J = gd.LR_gradient_descent(Xs2,ys,theta0)
    #theta2 , J2 = gd.LR_gradient_descent2(Xs2,ys,theta0)

    print(f'\ntheta_gradient: {theta}\n')
    #rint(theta2, J2)
    print(f'\n\ndifference: {theta_ridge-theta!s}')#' {np.mean(theta_ridge-theta)!s}\n')

    #print(f'{theta}')
