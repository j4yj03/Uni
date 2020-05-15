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
    X_m = multi.iloc[:,0:2].to_numpy()[:50]
    y_m = multi['y'].to_numpy()[:50]

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
    #theta0 = np.ones(n)
    theta0 = np.random.rand(n)

    print(f'Startvektor: {theta0}')

    theta_ridge = ridge.Ridge_fit(Xs2 ,ys, 0)

    print(f'\ntheta_normal:{theta_ridge}\n')


    thetas, costs, preds, counter = gd.LR_gradient_descent_hist(Xs2,ys,theta0, eta=0.01)

    theta = thetas[-1]

    print(f'\ntheta_gradient: {theta}\n nach {counter} Iterationen! \n')

    print(f'\n\ndifference: {theta_ridge-theta!s}')#' {np.mean(theta_ridge-theta)!s}\n')

    resolution = 0.0333

    fig = plt.figure(figsize=(11,9))

    X,Y = np.meshgrid(np.arange(-1, 1, resolution), np.arange(-1, 1, resolution))

    XX = X.flatten()
    YY = Y.flatten()

    #print(np.shape(XX),np.shape(YY))
    lst = np.array([YY, XX])
    #print(lst)
    # for x in XX:
    #     print(x)

    from itertools import chain, combinations_with_replacement

    combinations = list(chain.from_iterable(combinations_with_replacement(lst,i) for i in range(0, degree+1)))
    #leange der Liste entspricht ((m + degree)! / degree! + m! ) - 1
    YYXX=np.empty([len(lst[1]), len(combinations)])

    for ind, vector in enumerate(combinations):
        #Produkt der Kombinationsmöglichkeiten
        #print(ind, vector)
        if ind == 0:
            YYXX[:,0] = XX
            #pass
        else:
            YYXX[:,ind] = np.prod(vector, axis=0, dtype=np.double)

    #YYXX[-1] = theta
    error_mesh = []
    for th in YYXX:
        e = error(Xs2, ys, th)
        error_mesh.append(e)

    zs = np.array(error_mesh)

    print('theta meshgrid ',YYXX[np.where(zs == np.min(zs))], np.min(zs))

    zs_norm = (zs[:]-np.min(zs, axis=0))/np.ptp(zs, axis=0)

    Z = zs_norm.reshape(XX.shape)

    ax = Axes3D(fig)

    cset = ax.plot_surface(X, Y, Z.reshape(X.shape), rstride=1, cstride=1, alpha=0.5, cmap=cm.winter)

    fig.colorbar(cset, shrink=0.5, aspect=5)
    #cset = ax.scatter(X_m[:,0],X_m[:,1], y_m, c='r', edgecolors='black')
    ax.clabel(cset, fontsize=9, inline=1)

    costs_norm = (costs[:]-np.min(costs, axis=0))/np.ptp(zs, axis=0)
    for theta, cost in zip(thetas, costs):
        print(theta,cost)
    #print(thetas)
    ax.plot([t[0] for t in thetas], [t[1] for t in thetas], costs_norm   , marker='.', markersize=1)
    #ax.plot([t[0] for t in thetas], [t[1] for t in thetas], 0 ,marker='.', markersize=2)
    # for i in range(n):
    #     ax.plot([t[i] for t in thetas], [t[(i+1)%n] for t in thetas], costs_norm   , marker='.', markersize=2)

    ax.set_xlabel('theta0', labelpad=3, fontsize=12,)
    ax.set_ylabel('theta1', labelpad=3, fontsize=12,)
    ax.set_zlabel('cost', labelpad=3, fontsize=12)
    ax.view_init(elev=20., azim=30)
    #ax.set_title('X**{0:}:  alpha={1:.2E} -> R2={2:.4E}'.format(degree, df['alpha'].values[0], df['R2_score'].values[0]))
    plt.show()


# In[ ]:
