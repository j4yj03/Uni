###############################################################################
# Boston Dataset: lineare Regression
# Sidney Göhler 544131
#### IKT (M)
# Special Engineering SoSe20
# Prof. Dr. Andreas Zeiser
###############################################################################
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec
# import seaborn as sns #pip install searborn
import lineare_regression as lr
from sklearn.datasets import load_boston
from itertools import chain, combinations_with_replacement, product
#############################################
# Hilfsfunktionen
#############################################


def printhead(prefix="", array=None, lab=None):
    df = pd.DataFrame(array, columns=lab)
    print(prefix)
    print(train.head())
    print("shape: ", train.shape)
    print()


def columnNames(listToExtend, degree):
    list = []
    label = ""
    combinations = chain.from_iterable(
        combinations_with_replacement(listToExtend, i)
        for i in range(1, degree + 1))

    for combi in combinations:
        # print(combi)
        for c in combi:
            label = label + str(c)

        list.append(label)
        label = ""

    return list


def linear(degree=[1], alpha=[0], size=[100]):

    if not all(type(i) in (list, tuple, set) for i in [degree, alpha, size]):
        # if not type(degree) in (list, tuple, set, frozenset):
        print('WARNING: Hyperparameter muessen als Liste uebergeben werden.\n'
              + 'Verwende d=1 a=0 s=100.\n\n\n')
        degree = [1]
        alpha = [0]
        size = [100]

    r2_score_ridge = []
    mse_ridge = []
    cost = []
    theta_ridge = []
    fx = []
    legend = []

    hyperparameter = list(product(degree, alpha, size))
    hyperparameter = [[1,600,50],[2,43500,50]]


    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    #ax.ylabel('target')
    #ax.xlabel(features[column_to_plot])
    plt.xlim(min(X.T[column_to_plot]) - 0.2, max(X.T[column_to_plot]) + 0.2)

    for hypr in hyperparameter:
        d, a, s = hypr  # Polynomengrad, alpha und datensetgroesse

        X_tr = X_train[:s]
        y_tr = y_train[:s]
        X_te = X_test  # [:s]
        y_te = y_test  # [:s]

        print(f'###############\ncalculating and plotting for'
              + f' polynomialdegree = {d}, alpha = {a} and'
              + f' datasetsize = {np.shape(X_tr)}...\n')
        # print(_)
        X_tr_n = lr.QuadraticFeatures_fit_transform(X_tr, d)
        X_te_n = lr.QuadraticFeatures_fit_transform(X_te, d)

        # print(np.shape(X_tr),np.shape(X_tr_n),np.shape(y_tr),np.shape(a))

        theta = lr.Ridge_fit(X_tr_n, y_tr, a)
        theta_ridge.append(theta)
        #print("thetavactor: ",theta)
        print(f'theta calculated!\nwith {len(theta)} coefficients\n')

        #####
        # predict on linspace
        #####
        X_p = np.linspace(0, 10, 1000)
        if d == 1:
            y_pred_lin = lr.LR_predict(
                X_p, theta[column_to_plot:column_to_plot + 2])
        else:
            f = np.poly1d(np.flip(theta))
            fx.append(f)
            y_pred_lin = f(X_p)

        ######
        # predict on testdata
        ######

        y_pred_ridge = lr.Ridge_predict(X_te_n, theta)
        # print(y_pred_ridge)
        r2 = lr.r2_score(y_te, y_pred_ridge)
        r2_score_ridge.append(r2)
        # mse
        mse = lr.mean_squared_error(y_te, y_pred_ridge)
        mse_ridge.append(mse)
        print(
            f'validation on testdata '
            +f'({np.shape(X_te_n)}):\nr2 = {r2}\nmse = {mse}\n')
        # cost
        #cost.append(mse_ridge[i] + alpha[i] * np.sum(theta_ridge[i] ** 2, initial=1))

        print(
            f'plotting {features[column_to_plot]}'
            +f' ({column_to_plot}) traindata and predicted testdata now...')
        # plt.ylim(min(y_pred_ridge)-1,max(y_pred_ridge)+1)
        ax.plot([X_te.T[column_to_plot],
                  X_te.T[column_to_plot]], [y_pred_ridge, y_te],
                 color='0.75', linestyle='-.', linewidth=0.5)  # , label="mse")
        ax.scatter(X_tr.T[column_to_plot], y_tr, color='white',
                    edgecolors='black', label="training point")
        ax.scatter(X_te.T[column_to_plot], y_te, color='black',
                    edgecolors='white', label="test point")
        ax.scatter(X_te.T[column_to_plot], y_pred_ridge,
                    edgecolors='black', marker='x', label="predicted")

        ax2.scatter(y_pred_ridge, y_te)
        ax2.set_xlabel('y predicted')
        ax2.set_ylim(0,max(y)+1)
        ax2.set_ylabel('y')
        ax2.set_xlim(0,max(y)+1)
        ax2.plot(ax2.get_xlim(), ax2.get_ylim(), color='0.75', linestyle='dotted')
        ax2.set_title('R2 score = {}'.format(r2_score_ridge))
        ax2.grid(True)


        # plt.plot(X_p,y_pred_lin)

        # legend.append('training point')
        # legend.append('original unknown point')
        # legend.append('predicted point alpha={0:.2E}'.format(a))
        # legend.append('mse')
        # legend.append('mse')
        # plt.legend(numpoints=2)#legend, numpoints=1, bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        legend = []
        print('###############', '\n')
        #legend.append('alpha={0:.2E} -> R2={1:.4E}'.format(float(alpha[i]),float(r2_score_ridge[i])))



    plt.show()

#############################################
# Hyperparameter
#############################################


column_to_plot = 5  # RM

degree = [1, 2]
alpha = [600, 43500]
size = [50]


####################
# load datasets
####################
dataset = load_boston()
print(dataset.DESCR, '\n', '\n')
X = dataset.data[:-4]
y = dataset.target
features = dataset.feature_names


####################
# split dataset
####################

# for d in degree:
# X_tr_n = lr.QuadraticFeatures_fit_transform(X_tr, d)

X_train, X_test, y_train, y_test = lr.train_test_split(X, y, 0.6, 0)
X_test, X_val, y_test, y_val = lr.train_test_split(X_test, y_test, 0.5, 0)


#print(f"Shapes: \norginal = {np.shape(X)} \ntraindata = {np.shape(X_train)} \ntestdata = {np.shape(X_test)}: \nvalidationdata = {np.shape(X_val)}")


####################
# explore trainingset
####################

fig = plt.figure(figsize=(8, 8))
#sns.distplot(df_target, hist=True);
plt.hist(y_train)
plt.xlabel('target value')
# plt.ylim(0.5,2.2)
plt.ylabel('relative Häufigkeitsverteilung')
# plt.xlim(-0.5,11.)
plt.title('Histogramm der mittleren Hauspreise der Trainingsdaten')
plt.show()
# print(train_target.describe())

mean_train, std_train = [], []

#
# RM ist annähernd normal Verteilt un korreliert gut mit dem mittleren Hauspreis
# for ind, col in enumerate(X_train.T):
#     # print(df[col].describe())
#     mean, std = lr.StandardScaler_fit(col)
#     mean_train.append(mean)
#     std_train.append(std)
#     print(f'{features[ind]} ({ind}):')
#     fig1 = plt.figure(figsize=(16, 6))
#     gs = gridspec.GridSpec(1, 2)
#     ax = plt.subplot(gs[0])
#     ax.set_xlabel(features[ind])
#     ax.set_ylabel("relative Häufigkeitsverteilung")
#     ax.set_title("")
#     plt.hist(col)
#     # train[col].plot.density()
#     #sns.distplot(df[col], color='g')
#     ##
#     ax2 = plt.subplot(gs[1])
#     ax2.set_xlabel(features[ind])
#     ax2.set_ylabel("mittlerer Hauspreis")
#     plt.scatter(col, y_train)
#     ##
#     coeff = np.mean(np.corrcoef(col, y_train.T))
#     ax2.set_title("Korrelationskoeffizient: {0:.5f}".format(coeff))
#     plt.show()
#     print(f'rows = {np.size(col)}\nall numeric = {not np.isnan(col).any()} ({col.dtype})\nmean = {mean}\nstd = {std}', '\n', '\n')
#     # print(df[col].describe(),'\n','\n','\n','\n','\n','\n','\n')

linear(degree, alpha, size)

####################
# train on trainingset
####

####################
#
# #modellparameter berechnen
# theta_b = lreg.LR_fit(dataset.data, dataset.target.T)
# #bestimmtheitsmass ermitteln
# r2_score_b = lreg.r2_score(dataset.data, dataset.target.T, theta_b)
# #preise vorhersagen
# predicted_prize = lreg.LR_predict(dataset.data, theta_b)
#
# #vorhergesagt Preis über den eigentlichen Preis
# fig2 = plt.figure(figsize=(8,8))
# plt.scatter(predicted_prize,dataset.target.T)
# plt.xlabel('vorhergesagter Preis')
# plt.ylim(0,55)
# plt.ylabel('tatsächlicher Preis')
# plt.xlim(0,55)
# plt.plot(plt.xlim(), plt.ylim(),'--r')
# plt.title('Bestimmtheitsmaß {0:.5f}'.format(r2_score_b))
# plt.show()
