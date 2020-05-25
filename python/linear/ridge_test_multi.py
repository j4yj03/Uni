import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import ridge_regression as ridge
import lineare_regression as lin

from itertools import chain, combinations_with_replacement


def printhead(prefix = "", array=None, lab=None):
    df = pd.DataFrame(array, columns=lab)
    print(prefix)
    print(df.head())
    print("shape: ",df.shape)
    print()


def columnNames(listToExtend, degree):
    list = []
    label = ""
    combinations =  chain.from_iterable(combinations_with_replacement(listToExtend,i) for i in range(1, degree+1))

    for combi in combinations:
        #print(combi)
        for c in combi:
            label=label+str(c)

        list.append(label)
        label = ""

    return list



if __name__ == "__main__":

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None


    degree = 4

    #multi = pd.read_csv("./Uni/data/multivariat.csv", sep=',')
    multi = pd.read_csv("./data/multivariat.csv", sep=',')
    #multi = pd.read_csv("./data/univariat.csv", sep=',')
    #multi = multi.sort_values(by=['x'], ascending=True)
    printhead("Dataset",multi,list(multi.columns))
    multi_y = multi.pop('y')


    X_m = multi.to_numpy()#[:80]
    y_m = multi_y.to_numpy()#[:80]

    # m = 1000
    # X_m = 6*np.random.rand(m,1)-3
    # y_m = 0.5 * X_m**2 + X_m + 2 + np.random.randn(m,1)

    X_m_2 = ridge.QuadraticFeatures_fit_transform(X_m,degree)


    lab = columnNames(list(multi.columns),degree)

    printhead("Dataset mit Poly {}".format(degree),X_m_2 ,lab)
    #printhead(X_m_2,lab)

    X_m_train, X_m_test, y_m_train, y_m_test = ridge.train_test_split(X_m, y_m, 0.6, 0)
    X_m_test, X_m_val, y_m_test, y_m_val = ridge.train_test_split(X_m_test, y_m_test, 0.5, 0)
    X_m_2_train, X_m_2_test, y_m_2_train, y_m_2_test = ridge.train_test_split(X_m_2, y_m, 0.6, 0)
    X_m_2_test_2, X_m_2_val, y_m_2_test_2, y_m_2_val = ridge.train_test_split(X_m_2_test, y_m_2_test, 0.5, 0)

    #printhead("Trainingsdaten",X_m_2_train,lab)
    #printhead("Testdaten",X_m_2_test,lab)
    #printhead("Validierungsdaten",X_m_2_val,lab)


    legend=[]
    X= X_m_2_val
    y =y_m_2_val
    #plot

    #plt.plot(X_m_val, y_pred_lin)

    r2_score_ridge =[]
    mse_ridge = []
    cost = []
    theta_ridge = []
    fx = []
    alpha = []

    #print(X_m_2.shape)

    for i,a in enumerate([0.,4,40,20000,40000]):#
        alpha.append(a)
        #print(X_m_2.shape)
        #THETA berechnen
        theta_ridge.append(ridge.Ridge_fit(X_m_2_train,y_m_train, alpha[i]))
        #print("Parameter: ",theta_ridge[i]," test2: ", ridge.Ridge_fit2(X_m_2_train,y_m_train, alpha[i]))

        X_p = np.linspace(0,10,1000)
        #y_p = np.poly1d(np.flip(theta_ridge)[:,0])
        fx.append(np.poly1d(np.flip(theta_ridge[i])))
        #print(fx[i])
        #predict
        y_pred_ridge = ridge.Ridge_predict(X, theta_ridge[i])
        y_pred_lin = fx[i](X_p)

        r2_score_ridge.append(ridge.r2_score(y, y_pred_ridge))
        mse_ridge.append(ridge.mean_squared_error(y, y_pred_ridge))

        cost.append(mse_ridge[i] + alpha[i] * np.sum(theta_ridge[i] ** 2, initial=1))

        #plt.title('Bestimmtheitsmaß {0:.5f}'.format(r2_score_lin))

    data = {'theta': theta_ridge
            ,'mse':  mse_ridge
            ,'cost':  cost
            ,'R2_score':  r2_score_ridge
            ,'f(x)':  fx
            ,'alpha': alpha
            }
    #print(r2_score_ridge, mse_ridge, cost, theta_ridge ,fx)
    df = pd.DataFrame(data)
    df = df.nlargest(1,['R2_score'])
    print(df)

    # plot
    resolution = 0.25

    fig = plt.figure(figsize=(11,9))

    X,Y = np.meshgrid(np.arange(0, 10, resolution), np.arange(0, 10, resolution))
    #print(X.shape,Y.shape)

    XX = X.flatten()
    YY = Y.flatten()

    theta = df['theta'].values[0]
    print(theta)

    YYXX = ridge.QuadraticFeatures_fit_transform([XX, YY],degree)
    #print(YYXX.shape)

    Z = ridge.Ridge_predict(YYXX, theta)

    ax = Axes3D(fig)

    cset = ax.plot_surface(X, Y, Z.reshape(X.shape), rstride=1, cstride=1, alpha=0.7, cmap=cm.summer)

    fig.colorbar(cset, shrink=0.5, aspect=5)
    cset = ax.scatter(X_m[:,0],X_m[:,1], y_m, c='r', edgecolors='black')
    ax.clabel(cset, fontsize=9, inline=1)
    ax.set_title('X**{0:}:  alpha={1:.2E} -> R2={2:.4E}'.format(degree, df['alpha'].values[0], df['R2_score'].values[0]))
    plt.show()
