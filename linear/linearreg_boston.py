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

def plotlearningcurve(over,text):
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(2, 1)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax.set_title('RMSE bei verschiedenen {text}')
    ax.plot(over, np.sqrt(mse_test), label='testdata')
    ax.plot(over, np.sqrt(mse_train), label='trainingdata')
    ax.set_ylabel('root mse')
    ax.set_xlabel('{text}')
    ax.set_ylim(0,5)
    ax.set_xlim(min(over)-0.1,max(over)+0.1)
    ax.legend()
    ax2.set_title('RMSE bei verschiedenen {text} (vergroessert)')
    ax2.plot(over, np.sqrt(mse_test), label='testdata')
    ax2.plot(over, np.sqrt(mse_train), label='trainingdata')
    ax2.set_ylabel('root mse')
    ax2.set_xlabel('{text}')
    # ax2.set_xlim(110,160)
    # ax2.set_ylim(2.75,3.7)
    ax2.set_xlim(100,150)
    ax2.set_ylim(3.5,4.8)
    ax2.legend()
    plt.savefig('learncurve_{text}.png')
    plt.show()

def crossval(X_train, k):
    train_folds_score = []
    validation_folds_score = []
    alpha = np.logspace(-200,0.1, num=500, endpoint=True, base=2)

    for fold in range(0, k):

        size = X_train.shape[0]
        start = (size/k)*fold
        end = (size/k)*(fold+1)
        validation = X_train[start:end]


        training_set, validation_set = Cross_Validation.partition(examples, fold, k)
        training_labels, validation_labels = Cross_Validation.partition(labels, fold, k)

        learner.fit(training_set, training_labels)
        training_predicted = learner.predict(training_set)
        validation_predicted = learner.predict(validation_set)


    return train_folds_score, validation_folds_score

def linear(degree=[1], alpha=[0], size=[100]):

    if not all(type(i) in (list, tuple, set) for i in [degree, alpha, size]):
        # if not type(degree) in (list, tuple, set, frozenset):
        print('WARNING: Hyperparameter muessen als Liste uebergeben werden.\n'
              + 'Verwende d=1 a=0 s=100.\n\n\n')
        degree = [1]
        alpha = [0]
        size = [100]

    r2_score_ridge = []
    mse_test = []
    mse_train = []
    cost = []
    theta_ridge = []
    fx = []
    legend = []
    #12345
    hyperparameter = list(product(degree, alpha, size))
    #hyperparameter =[[1,0.01,405],[2,0.01,405]]# [[1,600,200],[2,45339,405]]#,[3,8000000,200]]


    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0])

    ax2 = plt.subplot(gs[1])


    for hypr in hyperparameter:
        d, a, s = hypr  # Polynomengrad, alpha und datensetgroesse

        X_tr = X_train[:s]
        y_tr = y_train[:s]
        X_te = X_test#[:s]
        y_te = y_test#[:s]

        print(f'###############\ncalculating and plotting for:\n'
              + f'polynomialdegree = {d}\nalpha = {a}\n'
              + f'datasetsize = {np.shape(X_tr)}...\n')
        # print(_)
        X_tr_n = lr.QuadraticFeatures_fit_transform(X_tr, d)
        X_te_n = lr.QuadraticFeatures_fit_transform(X_te, d)

        # print(np.shape(X_tr),np.shape(X_tr_n),np.shape(y_tr),np.shape(a))

        theta = lr.Ridge_fit(X_tr_n, y_tr, a)
        theta_ridge.append(theta)
        #print("thetavactor: ",theta)
        print(f'theta calculated!\nwith {len(theta)} coefficients\n')

        ######
        # predict on testdata
        ######

        y_pred_test = lr.Ridge_predict(X_te_n, theta)
        y_pred_train = lr.Ridge_predict(X_tr_n, theta)
        # print(y_pred_test)
        r2 = lr.r2_score(y_te, y_pred_test)
        r2_score_ridge.append(r2)
        # mse
        mse = lr.mean_squared_error(y_te, y_pred_test)
        mse2 = lr.mean_squared_error(y_tr, y_pred_train)
        mse_test.append(mse)
        mse_train.append(mse2)
        print(
            f'validation on testdata '
            +f'({np.shape(X_te_n)}):\nr2 = {r2}\nmse = {mse}\n')


        #######################################################################
        #plot mse
        ax.plot([X_te.T[column_to_plot],
                  X_te.T[column_to_plot]], [y_pred_test, y_te],
                 color='0.75', linestyle='dotted', linewidth=0.5)

        ax.scatter(X_te.T[column_to_plot], y_pred_test,
                    edgecolors='black', marker='x', label=f'degree = {d}\nalpha = {a}\nmse = {mse:.5f}')

        ax2.scatter(y_pred_test, y_te, marker='.', label=f'degree = {d}\nalpha = {a}\nr2 = {r2:.5f}')




    print('###############', '\n')
    print(
        f'plotting {features[column_to_plot]}'
        +f' ({column_to_plot}) traindata and predicted testdata now...')
        #legend.append('alpha={0:.2E} -> R2={1:.4E}'.format(float(alpha[i]),float(r2_score_ridge[i])))
    legend.append('training point')
    legend.append('original unknown point')

    ax.set_title(f'{features[column_to_plot]}: Trainingspunkte m_tr = {len(X_tr)} sowie Testpunkte m_t = {len(X_te)}')
    #ax.set_xlim(min(X.T[column_to_plot]) - 0.2, max(X.T[column_to_plot]) + 0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax2.set_title('Residuum')
    ax2.set_xlabel('y predicted')
    ax2.set_ylim(0, ax.get_ylim()[1])
    ax2.set_ylabel('y')
    ax2.set_xlim(0,ax.get_ylim()[1])

    ax.scatter(X_tr.T[column_to_plot], y_tr, color='white',
                edgecolors='black', label="training point")
    ax.scatter(X_te.T[column_to_plot], y_te, color='black',
                marker='.', label="test point")

    ax2.plot(ax2.get_xlim(), ax2.get_ylim(), color='0.1', linestyle='-.', linewidth=0.5, label='r2 = 1')
    #fig.savefig(f'./pics/linearreg_{features[column_to_plot]}_{hyperparameter}_{np.shape(X_te)}.png', bbox_inches='tight')
    ax.legend()
    ax2.legend()
    ax2.grid(color='0.7', linestyle='-.', linewidth=0.5)
    plt.show()

    return hyperparameter, r2_score_ridge, mse_test, mse_train, cost, theta_ridge

#############################################
# Hyperparameter
#############################################




# degree = [1]
# alpha = list(np.arange(0.0001,0.5,0.001)) list(np.logspace(-200,0.1, num=500, endpoint=True, base=2))
# size = [100]

column_to_plot = 5

# degree = [1]
# alpha = [600]
# size = list(range(1,405,1))
degree = [1,2]
alpha = [0.0035,0.035,0.35]
size = [4000]


print(alpha)
####################
# load datasets
####################
dataset = load_boston()
print(dataset.DESCR, '\n', '\n')
X = dataset.data#[:-4]
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

# Häufigkeitsverteilung y (target)
fig = plt.figure(figsize=(9, 8))
plt.hist(y_train)
plt.xlabel('target value')
plt.ylabel('relative Häufigkeitsverteilung')
plt.title('Histogramm der mittleren Hauspreise der Trainingsdaten')
plt.show()

mean_train, std_train = [], []
coeffs = []

#
for ind, col in enumerate(X_train.T):
    # print(df[col].describe())
    mean, std = lr.StandardScaler_fit(col)
    mean_train.append(mean)
    std_train.append(std)
    print(f'{features[ind]} ({ind}):')
    fig1 = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0])
    ax.set_xlabel(features[ind])
    ax.set_ylabel("relative Häufigkeitsverteilung")
    ax.set_title("")
    plt.hist(col)
    ##
    ##
    ax2 = plt.subplot(gs[1])
    ax2.set_xlabel(features[ind])
    ax2.set_ylabel("mittlerer Hauspreis")
    plt.scatter(col, y_train)
    ##
    coeff = np.mean(np.corrcoef(col, y_train.T))
    coeffs.append(coeff)
    ax2.set_title("Korrelationskoeffizient: {0:.5f}".format(coeff))
    plt.show()

    #Feature beschreiben
    print(f'rows = {np.size(col)}\nall numeric = {not np.isnan(col).any()} ({col.dtype})\nmean = {mean}\nstd = {std}\ncorrcoef = {coeff}', '\n', '\n')

# coeffinds = np.array(coeffs).argsort()
#
# X_sorted = X.T[coeffinds[::-1]]
# features_sorted = features[coeffinds[::-1]]




# Datenset mit entfernten Features
X = dataset.data
X = X[:5000]
y = y[:5000]
# mask = np.ones(np.size(dataset.data,1), dtype=bool)
# mask[[-4,-2]] = False
# X = dataset.data[:,mask]

## Datenset erneut aufsplitten
X_train, X_test, y_train, y_test = lr.train_test_split(X, y, 0.6, 0)
X_test, X_val, y_test, y_val = lr.train_test_split(X_test, y_test, 0.5, 0)
X_train = np.vstack((X_train, X_val))
y_train = np.hstack((y_train, y_val))

print(X_train.shape,y_train.shape)

hyperparameter, r2_score_ridge, mse_test, mse_train, cost, theta_ridge = linear(degree, alpha, size)


print('')

over = size
text = 'size'
plotlearningcurve(over, text)



# ##############################################################################################
# for _ in range(np.size(X,1)):
#     column_to_plot = _
#     hyperparameter, r2_score_ridge, mse_test, cost, theta_ridge = linear(degree, alpha, size)
#
#
#
#
# r2_score_ridgeinds = np.array(r2_score_ridge).argsort()
# hyperparameter_sorted = hyperparameter[r2_score_ridgeinds[-1]]
# print(F'best hyperparameters = {hyperparameter_sorted}')


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
