import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import linearreg as lreg



if __name__ == "__main__":

    #univariat
    uni = pd.read_csv("./lineare_regression_daten/univariat.csv", sep=',')
    x_u = uni['x'].to_numpy()
    y_u = uni['y'].to_numpy()

    #multivariat
    multi = pd.read_csv("./lineare_regression_daten/multivariat.csv", sep=',')
    #x1_m = multi['x1'].to_numpy()
    #x2_m = multi['x2'].to_numpy()
    X_m = multi.iloc[:,0:2].to_numpy()
    y_m = multi['y'].to_numpy()

    #print(X_m.shape,y_m.shape)
    #print(x2,x2.shape)
    #theta_u = lreg.LR_fit(x_u, y_u)
    #predict_u = lreg.LR_predict(np.linspace(-1,10,10), theta_u)
    theta_m = lreg.LR_fit(X_m, y_m)
    r2_score_m = lreg.r2_score(X_m, y_m, theta_m)
    predict_m = lreg.LR_predict(X_m, theta_m)


    # fig = plt.figure(figsize=(8,8))
    # plt.scatter(x_u, y_u)
    # plt.plot(predict_u,'r')
    # plt.xlabel('x')
    # plt.ylim(0.5,2.2)
    # plt.ylabel('y')
    # plt.xlim(-0.5,11.)
    # plt.show()


    fig = plt.figure(figsize=(8,8))
    t_eval = np.linspace(0,10,10)
    XX1,XX2 = np.meshgrid(t_eval,t_eval)
    X_eval = np.concatenate((XX1.reshape(-1,1),XX2.reshape(-1,1)),axis=1)
    y_eval = lreg.LR_predict(X_eval,theta_m)
    YY = y_eval.reshape(XX1.shape)
    ax = Axes3D(fig)
    cset = ax.plot_surface(XX1,XX2,YY,cmap=cm.coolwarm)
    cset = ax.scatter(X_m[:,0],X_m[:,1],y_m, c='g')
    ax.clabel(cset, fontsize=9, inline=1)
    ax.set_title('Bestimmtheitsmaß {0:.5f}'.format(r2_score_m))
    plt.show()
