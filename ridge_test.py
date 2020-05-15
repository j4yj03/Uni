import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        print(combi)
        for c in combi:
            label=label+str(c)

        list.append(label)
        label = ""

    return list



if __name__ == "__main__":

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None


    degree = 2
    alpha = 0

    #multi = pd.read_csv("./Uni/data/multivariat.csv", sep=',')
    #multi = pd.read_csv("./data/multivariat.csv", sep=',')
    multi = pd.read_csv("./data/univariat.csv", sep=',')
    #multi = multi.sort_values(by=['x'], ascending=True)
    printhead("Dataset",multi,list(multi.columns))

    multi_y = multi.pop('y')
    X_m = multi.to_numpy()[:60]
    y_m = multi_y.to_numpy()[:600]

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
    X_m_2_test, X_m_2_val, y_m_2_test, y_m_2_val = ridge.train_test_split(X_m_2_test, y_m_2_test, 0.5, 0)

    printhead("Trainingsdaten",X_m_2_train,lab)
    printhead("Testdaten",X_m_2_test,lab)
    printhead("Validierungsdaten",X_m_2_val,lab)

    legend=[]
    X= X_m_2_val
    y =y_m_2_val
    #plot
    fig = plt.figure(figsize=(11,9))

    plt.xlabel('x')
    plt.ylim(min(y_m_train)-0.05,max(y_m_train)+0.05)
    plt.ylabel('y')
    plt.xlim(min(X_m_train)-0.05,max(X_m_train)+0.05)
    plt.scatter(X_m_train, y_m_train, c='white', edgecolors='black')

    #plt.plot(X_m_val, y_pred_lin)
    r2_score_ridge =[]
    mse_ridge = []
    cost = []
    theta_ridge = []
    fx = []
    alpha = []

    for i,a in enumerate([0., 0.05, 0.5, 5]):
        alpha.append(a)
        #THETA berechnen

        theta_ridge.append(ridge.Ridge_fit(X_m_2_train,y_m_train, alpha[i]))
        print("Parameter: ",theta_ridge[i]," test2: ",ridge.Ridge_fit2(X_m_2_train,y_m_train, alpha[i]))

        X_p = np.linspace(0,10,1000)
        fx.append(np.poly1d(np.flip(theta_ridge[i])))
        #print(fx[i])
        y_pred_ridge = ridge.Ridge_predict(X, theta_ridge[i])
        y_pred_lin = fx[i](X_p)
        #y_pred_ridge = ridge.Ridge_predict(X_m_2_val, theta_ridge)

        # for x,y in zip(X_m_2_val,y_pred_ridge):
        #     print("x={} y={}".format(x,y))

        # r2 score
        r2_score_ridge.append(ridge.r2_score(y, y_pred_ridge))
        # mse
        mse_ridge.append(ridge.mean_squared_error(y, y_pred_ridge))
        # cost
        cost.append(mse_ridge[i] + alpha[i] * np.sum(theta_ridge[i] ** 2, initial=1))


        plt.scatter(X[:,0], y, color='black', edgecolors='white')
        plt.scatter(X[:,0], y_pred_ridge, edgecolors='black')
        plt.plot(X_p,y_pred_lin)
        legend.append('alpha={0:.2E} -> R2={1:.4E}'.format(float(alpha[i]),float(r2_score_ridge[i])))

        #endloop

    data = {'theta': theta_ridge
            ,'mse':  mse_ridge
            ,'cost':  cost
            ,'R2_score':  r2_score_ridge
            ,'f(x)':  fx
            ,'alpha': alpha
            }

    df = pd.DataFrame(data)
    df = df.nlargest(1,['R2_score'])
    print("\n\n f(x) = {}\n\n with R2 Score = {}\n\n".format(df['f(x)'].values[0],df['R2_score'].values[0]))


    plt.title('theta_best = {}'.format(df['theta'].values[0]))
    legend.append('training point')
    legend.append('unknown point')
    plt.legend(legend,bbox_to_anchor=(0, 1), loc='upper left', ncol=1)

    plt.show()

    #endscript
