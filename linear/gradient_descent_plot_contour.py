#!/usr/bin/env python
# coding: utf-8

# In[19]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#import lineare_regression as lreg
import gradient_descent as gd
import ridge_regression as ridge

def error(X, Y, THETA):

    loss = ridge.Ridge_predict(X, THETA) - Y

    #print(X.shape, Y.shape, y_pred.shape, THETA.shape)
    return np.mean(loss**2)

# In[91]:


degree = 2
anzahlds = 10

# In[92]:


if __name__ == "__main__":

    #univariat
    #uni = pd.read_csv("./data/univariat.csv", sep=',')
    #x_u = uni['x'].to_numpy()
    #y_u = uni['y'].to_numpy()

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


    mean, std = gd.StandardScaler_fit(X_m)
    Xs = gd.StandardScaler_transform(X_m, mean, std)
    Xm2 = ridge.QuadraticFeatures_fit_transform(X_m, degree)
    Xs2 = ridge.QuadraticFeatures_fit_transform(Xs, degree)

    print(f'Xm: mean={mean} std={std}')

    m = np.size(X_m,1)
    n = (np.floor_divide(np.math.factorial(m + degree),
                np.math.factorial(degree) *
                    np.math.factorial(m)))


    # theta0 initialisieren
    #theta0 = np.zeros(n)
    theta0 = np.ones(n)
    #theta0 = np.random.rand(n)

    print(f'Startvektor: {theta0}')


    # theta per normalengleichung
    theta_ridge = ridge.Ridge_fit(Xs2 ,ys, 0)

    print(f'\ntheta_normal:{theta_ridge}\n mse: {ridge.mean_squared_error(ys, ridge.Ridge_predict(Xs2, theta_ridge))} ')


    thetas, costs, preds, counter = gd.LR_gradient_descent_hist(Xs2[:anzahlds],ys[:anzahlds],theta0, eta=0.01)

    theta = thetas[-1]

    print(f'\ntheta_gradient: {theta}\n mse: {ridge.mean_squared_error(ys, ridge.Ridge_predict(Xs2, theta))}\n  nach {counter} Iterationen! ')

    print(f'\ndifference: {theta_ridge-theta!s}')#' {np.mean(theta_ridge-theta)!s}\n')

    resolution = 0.5
    #print(len(thetas))
    fig = plt.figure(figsize=(11,9))

    X,Y = np.meshgrid(np.arange(-1, 1, resolution/10), np.arange(-1, 1, resolution/10))

    XX = np.arange(-1, 1, resolution)
    YY = Y.flatten()

    print(np.shape(XX),np.shape(YY))
    #lst = np.array([XX,YY])
    #for i, x in enumerate(XX):
    #    print(i,x)

    import itertools

    #combinations = chain.from_iterable(combinations_with_replacement(X,i) for i in range(1, degree+1))
    #leange der Liste entspricht ((m + degree)! / degree! + m! ) - 1
    YYXX=[]
    error_mesh = []
    for vector in itertools.combinations_with_replacement(XX, n):
        #vector = [-0.06777297, -0.603243,    0.26951983,  0.05849988, -0.08877793,  0.01141836]
        e = error(Xs2, ys, vector)
        error_mesh.append(e)
    print(len(error_mesh))
    #np.random.shuffle(YYXX)
    #YYXX[-1] = theta




    zs = np.array(error_mesh)
    print(np.min(zs))
    #print('theta meshgrid ',YYXX[np.where(zs == np.min(zs))], np.min(zs))

    zs_norm = zs# (zs[:]-np.min(zs, axis=0))/np.ptp(zs, axis=0)

    Z = zs_norm.reshape(X.shape)

    #ax = Axes3D(fig)
    print(f'{Z.shape}{X.shape}{Y.shape}')

    #Z_re
    print(X[np.where(Z==np.min(Z))], Y[np.where(Z==np.min(Z))])
    cp = plt.contour(X, Y, Z, np.logspace(-30,np.max(error_mesh)+4,22,base=1.15), colors='black', linestyles='dashed', linewidths=1)
    plt.clabel(cp, inline=1, fontsize=10)
    cp = plt.contourf(X, Y, Z, np.logspace(-30,np.max(error_mesh)+4,22,base=1.15))
    plt.plot(X[np.where(Z==np.min(Z))], Y[np.where(Z==np.min(Z))], color='white', marker='x', markersize=3)
    plt.annotate(f'{np.around(np.min(Z),5)}', xy=(X[np.where(Z==np.min(Z))]+0.025, Y[np.where(Z==np.min(Z))]), color='white')
    #cset = ax.plot_surface(X, Y, Z.reshape(X.shape), rstride=1, cstride=1, alpha=0.5, cmap=cm.winter)
    #ax.plot(X[np.where(np.min(zs))][0], Y[np.where(np.min(zs))][0], np.min(zs_norm), marker='.', markersize=3)

    #fig.colorbar(cset, shrink=0.5, aspect=5)
    #cset = ax.scatter(X_m[:,0],X_m[:,1], y_m, c='r', edgecolors='black')
    #ax.clabel(cset, fontsize=9, inline=1)

    costs_norm = costs #(costs-np.min(costs, axis=0))/np.ptp(zs, axis=0)
    print(costs_norm)

    # for theta, cost in zip([thetas[-1]], [costs[-1]]):
    #      print(theta,cost)
    #print(thetas)
    #plt.plot([t[1] for t in [thetas[-1]]], [costs_norm[-1]] , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=7)
    #plt.plot([t[1] for t in thetas], [t[2] for t in thetas], color='r')
    plt.annotate(f'{[costs[-1]]!s}', xy=(0, 0))

    #ax.plot([t[1] for t in thetas], [t[2] for t in thetas], 0 ,marker='.', markersize=2)
    for i in range(n):
        plt.plot([t[i] for t in thetas], [t[(i+1)%n] for t in thetas])

    plt.xlabel('theta0', labelpad=3, fontsize=12,)
    plt.ylabel('theta1', labelpad=3, fontsize=12,)
    #ax.set_zlabel('cost', labelpad=3, fontsize=12)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    #ax.set_zlim(0,np.max(costs_norm)+0.05)
    #ax.view_init(elev=0., azim=180)
    #ax.set_title('X**{0:}:  alpha={1:.2E} -> R2={2:.4E}'.format(degree, df['alpha'].values[0], df['R2_score'].values[0]))
    plt.show()


# In[ ]:
