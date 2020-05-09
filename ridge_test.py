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
    list = listToExtend
    label = ""
    combinations =  chain.from_iterable(combinations_with_replacement(listToExtend,i) for i in range(degree, degree+1))
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


    degree = 2
    alpha = 100

    multi = pd.read_csv("./Uni/data/multivariat.csv", sep=',')
    printhead("Dataset",multi,list(multi.columns))
    multi_y = multi.pop('y')



    X_m = multi.to_numpy()
    X_m_2 = ridge.QuadraticFeatures_fit_transform(X_m,degree)
    y_m = multi_y.to_numpy()

    lab = columnNames(list(multi.columns),degree)

    #printhead(X_m_2,lab)

    X_m_train, X_m_test, y_m_train, y_m_test = ridge.train_test_split(X_m, y_m, 0.6, 0)
    X_m_test, X_m_val, y_m_test, y_m_val = ridge.train_test_split(X_m_test, y_m_test, 0.5, 0)
    X_m_2_train, X_m_2_test, y_m_2_train, y_m_2_test = ridge.train_test_split(X_m_2, y_m, 0.6, 0)
    X_m_2_test, X_m_2_val, y_m_2_test, y_m_2_val = ridge.train_test_split(X_m_2_test, y_m_2_test, 0.5, 0)

    printhead("Trainingsdaten",X_m_2_train,lab)
    printhead("Testdaten",X_m_2_test,lab)
    printhead("Validierungsdaten",X_m_2_val,lab)


    #THETA berechnen
    theta_lin = lin.LR_fit(X_m_train, y_m_train)
    theta_ridge = ridge.Ridge_fit(X_m_2_train, y_m_2_train, alpha)
    print("theta lin: ",theta_lin," | theta ridge: ",theta_ridge)


    #prediction
    y_pred_ridge = ridge.Ridge_predict(X_m_2_val, theta_ridge)
    y_pred_lin = lin.LR_predict(X_m_val, theta_lin)




    #r2 score
    r2_score_lin = lin.r2_score(X_m_val, y_m_val, theta_lin)
    r2_score_ridge = lin.r2_score(X_m_2_val, y_m_2_val, theta_ridge)
    print("r2 lin: ",r2_score_lin," | r2 ridge: ", r2_score_ridge)

    #mse
    mse_lin = ridge.mean_squared_error(y_m_val, y_pred_lin)
    mse_ridge = ridge.mean_squared_error(y_m_2_val, y_pred_ridge)
    print("mse lin: ",mse_lin," | mse ridge: ", mse_ridge)

    # fig = plt.figure(figsize=(8,8))
    # t_eval = np.linspace(0,10,10)
    # print(t_eval.shape)
    # XX1,XX2,XX3,XX4,XX5 = np.meshgrid(t_eval,t_eval,t_eval,t_eval,t_eval)
    # X_eval = np.concatenate((XX1.reshape(-1,1),XX2.reshape(-1,1),XX3.reshape(-1,1),XX4.reshape(-1,1),XX5.reshape(-1,1)),axis=1)
    # print(XX1.shape)
    # print(theta_ridge.shape)
    # y_eval = ridge.Ridge_predict(X_eval ,theta_ridge)
    #
    # print(y_eval)
    # YY = y_eval.reshape(XX1.shape)
    # ax = Axes3D(fig)
    # print(YY.shape)
    #
    # cset = ax.plot_surface(XX1,XX2,XX3,XX4,XX5,YY,cmap=cm.coolwarm)
    # cset = ax.scatter(X_m_train[:,0],X_m_train[:,1],y_m_train, c='g')
    # ax.clabel(cset, fontsize=9, inline=1)
    # #ax.set_title('Bestimmtheitsmaß {0:.5f}'.format(r2_score_m))
    # plt.show()

    #df = pd.DataFrame(X_m2, columns=lab)

    #print(df.head())
