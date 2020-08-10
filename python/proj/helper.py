import pandas as pd
import numpy as np

from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.stats import skew

import time
from sys import getsizeof
import dill
from itertools import permutations, count #,izip

from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg


from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from keras.layers import Input, Dense, Dropout, LSTM, Conv2D, MaxPooling2D, UpSampling2D
from keras import regularizers
from keras.losses import BinaryCrossentropy, mean_squared_error, KLDivergence# SparseCategoricalCrossentropy
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping




#######################################################################################################################
#######################################################################################################################
def plot_stats(dr, axis=0, color='tab:blue', plot=False):
        
    a = (np.std(dr, axis=axis))
    b = (np.var(dr, axis=axis))
    c = (skew(dr, axis=axis))
    d = (kurtosis(dr, axis=axis))
    
    if plot:
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)

        ax.set_title('Histogramm')
        sns.distplot(dr,  ax=ax, color=color);
        #ax.legend((*hist.legend_elements()))
        plt.show()
    
    
        fig = plt.figure(figsize=(14,14))
        ax1 = fig.add_subplot(221)
        ax1.plot(a, 'tab:red', label='std')
        ax1.legend()
        ax2 = fig.add_subplot(222)
        ax2.plot(b, 'tab:green', label='var')
        ax2.legend()
        ax3 = fig.add_subplot(223)
        ax3.plot(d, 'tab:purple', label='kurtosis')
        ax3.legend()
        ax4 = fig.add_subplot(224)
        ax4.plot(c, 'tab:olive', label='skew')
        ax4.legend()
        plt.show()
    
    return a, b, c, d
#######################################################################################################################
#######################################################################################################################
def clust_eval(y_pred, y_train, name=""):
    fig1 = plt.figure(figsize=(16,6))
    gs = gridspec.GridSpec(1,2)

    ax1 = plt.subplot(gs[0])
    cnf_matrix = confusion_matrix(y_train,y_pred)
    row_sum = cnf_matrix.sum(axis=1,keepdims=True)
    #print(cnf_matrix)
    #print(row_sum)
    cnf_matrix_norm =cnf_matrix / row_sum
    sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True, xticklabels=["1 good","0 defect"], yticklabels=["1 good","0 defect"])

    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    plt.title(f"Normalized Confusion Matrix - {name}")


    #ROC Curve Trainingdata 



    ax3 = plt.subplot(gs[1])

    fpr, tpr, thresholds = roc_curve(y_train, y_pred)

    roc_auc = auc(fpr, tpr)

    
    plt.plot(fpr, tpr, lw=1, alpha=0.9,
             label='ROC (AUC = %0.2f)' % (roc_auc))


    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.7)


    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    
    return roc_auc
#######################################################################################################################
#######################################################################################################################
def eval_gmm(model, good, goodtrain, goodtest, goodvalid, defect, thres):

    y_pred= np.ones(len(defect))
    y_pred_g= np.ones(len(good))

    # predict on train and estimate treshold
    densities = model.score_samples(goodtrain)
    density_threshold = np.percentile(densities, thres)

    outliers = goodtrain[densities >= density_threshold]


    print('traindata:',len(outliers))


    # predict on testdata
    densities = model.score_samples(goodtest)
    outliers = goodtest[densities >= density_threshold]


    print('testdata:',len(outliers))


    # predict on validation 
    densities = model.score_samples(goodvalid)
    outliers = goodvalid[densities >= density_threshold]


    print('validationsdata:',len(outliers))



    # predict on good Lines
    densities = model.score_samples(good)
    outliers = good[densities >= density_threshold]


    print('######################\nestimated on Good Lines DS:',len(outliers),'\n\n')
    y_pred_g[outliers.index] = 0
    
    
    # estimate density of goodlines
    densities = model.score_samples(good)
    density_threshold = np.percentile(densities, thres)
    # predict on defect Lines
    densities_defect = model.score_samples(defect)
    
    
    outliers_defect = defect[densities_defect >= density_threshold]


    print('######################\nestimated on Defect Lines DS:',len(outliers_defect))
    y_pred[outliers_defect.index] = 0
    
    return y_pred, y_pred_g
#######################################################################################################################
#######################################################################################################################
def ae_classy(mse, threshold):
    y_pred_ae = []
    
    for i, loss in enumerate(mse):

        if loss >= threshold:
            y_pred_ae.append(0) # defect
        else:
            y_pred_ae.append(1) # good
            
    return y_pred_ae
#######################################################################################################################
#######################################################################################################################
def ae_reconstruct(lab, title, sets, model):

    cmaps = [cm.Dark2, cm.Accent]

    for t, set_, colorm in zip(title, sets, cmaps):

        fig = plt.figure(figsize=(18,6))
        plt.ylabel('loss')
        plt.xlabel('data index')
        plt.grid(linestyle=':', linewidth=0.5)
        
        #plt.ylim(0,0.8)
        plt.title(f'autoencoder reconstruction error: {t}')

        for ind, l, s in zip(count(), lab, set_):

            predictions = model.predict(s)
            loss = np.mean(np.power(s - predictions, 2), axis=1)

            color = colorm(ind)

            error_df = pd.DataFrame({'reconstruction_error': loss})
            error_df.sort_index(inplace=True)
            error_df.reset_index(inplace=True)

            plt.plot(error_df['reconstruction_error'], label=f'{l}', alpha=0.9, color=color)

        plt.legend()
        plt.show()
#######################################################################################################################        
#######################################################################################################################        
def ae_reconstruct_fullset(lab, sets, model):
    losses = []
    ticks = range(0,len(sets[0]),1000)


    fig = plt.figure(figsize=(18,6))
    plt.xlim([-10,17099])
    plt.xticks(ticks)
    plt.ylabel('loss')
    plt.xlabel('data index')
    plt.grid(linestyle=':', linewidth=0.5)

    plt.title('autoencoder reconstruction error')

    for ind, d in enumerate(sets):

        predictions = model.predict(d)
        loss = np.mean(np.power(d - predictions, 2), axis=1)

        color = cm.Dark2(ind)

        error_df = pd.DataFrame({'reconstruction_error': loss})

        error_df.sort_index(inplace=True)
        error_df.reset_index(inplace=True)

        plt.plot(error_df['reconstruction_error'], label=f'{lab[ind]}', color=color)
        losses.append(error_df['reconstruction_error'].to_numpy())

    loss = error_df[1:]['reconstruction_error']

    threshold = np.max(loss)-(np.max(loss)/5)


    plt.axhline(threshold,-1,17100, color='tab:grey', linestyle='--', lw=2, alpha=0.5, label='threshold')

    plt.legend()
    plt.show()
    
    return threshold, losses
#######################################################################################################################
#######################################################################################################################
def create_model(loss_fn = "mse", input_dim=2, patience=10, file="autoencoder.h5"):    
    
    input_layer = Input(shape=(input_dim, ))

    encoder = Dense(int(input_dim / 2), activation="relu")(input_layer)

    dropout = Dropout(0.2)(encoder)

    encoder = Dense(int(input_dim / 4), activation="relu")(dropout)

    dropout = Dropout(0.2)(encoder)

    decoder = Dense(int(input_dim / 2), activation='relu')(dropout)

    decoder = Dense(input_dim, activation=None)(decoder)

    checkpointer = ModelCheckpoint(filepath=file,
                           verbose=0,
                           save_best_only=True)

    early_stopping = EarlyStopping(patience=patience,
                           verbose=0,
                           restore_best_weights=True)

    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    autoencoder.compile(optimizer='adam', 
                    loss=loss_fn,
                       metrics=['accuracy'])
    #autoencoder.compile(optimizer='adadelta',
#                    loss='binary_crossentropy', 
#                    metrics=['accuracy'])

    
    return autoencoder, checkpointer, early_stopping
#######################################################################################################################
#######################################################################################################################
def setup_ae_and_train(nb_epoch, batch_size, patience, input_dim, trainset, validset):

    loss_fn = "mse"#KLDivergence()# mean_squared_error# "mse"# MeanSquaredError()# BinaryCrossentropy() loss='kld'

    file = Path(f'modelle//autoencoder_transposed_{time.time()}.h5').absolute().as_posix()

    nb_epoch = 30
    batch_size = 100
    patience = 10


    autoencoder, checkpointer, early_stopping = create_model(loss_fn, input_dim, patience, file)

    autoencoder.summary()
    
    print(f'\nsaving model to {file}\n')
    history = autoencoder.fit(trainset, trainset,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,                                                                 
                    validation_data=(validset, validset),
                    verbose=1,
                    callbacks=[checkpointer, early_stopping])
    
    fig = plt.figure(figsize=(16,10))
    df = pd.DataFrame(history.history)

    #plt.yticks(ticks)
    #plt.ylim(0.1,1.05)
    plt.xlabel('epochs')
    plt.plot(df)
    plt.legend(df.columns)
    plt.show()

    print('min val loss:',min(history.history['val_loss']))
    
    return autoencoder, history
