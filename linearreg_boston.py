import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec
import seaborn as sns #pip install searborn

import linearreg as lreg

from sklearn.datasets import load_boston


#print(dataset.DESCR)
if __name__ == "__main__":
    #print(dataset.target)
    dataset = load_boston()

    #dataset info
    print(dataset.DESCR)

    #Histogramm über den mittleren Hauspreis
    fig = plt.figure(figsize=(8,8))
    sns.distplot(dataset.target, hist=True);
    plt.xlabel('x')
    #plt.ylim(0.5,2.2)
    plt.ylabel('relative Häufigkeitsverteilung')
    #plt.xlim(-0.5,11.)
    plt.title('Histogramm der mittleren Hauspreise')
    plt.show()
    #

    # RM ist annähernd normal Verteilt un korreliert gut mit dem mittleren Hauspreis
    for ind, name in enumerate(dataset.feature_names):
        fig1 = plt.figure(figsize=(16,6))
        gs = gridspec.GridSpec(1,2)
        ax = plt.subplot(gs[0])
        ax.set_xlabel(name)
        ax.set_ylabel("relative Häufigkeitsverteilung")
        ax.set_title("")
        sns.distplot(dataset.data.T[ind], color='g')
        ##
        ax2 = plt.subplot(gs[1])
        ax2.set_xlabel(name)
        ax2.set_ylabel("mittlerer Hauspreis")
        plt.scatter(dataset.data.T[ind], dataset.target.T)
        ##
        coeff = np.mean(np.corrcoef(dataset.data.T[ind], dataset.target.T))
        ax2.set_title("Korrelationskoeffizient: {0:.5f}".format(coeff))
        plt.show()

    #modellparameter berechnen
    theta_b = lreg.LR_fit(dataset.data, dataset.target.T)
    #bestimmtheitsmass ermitteln
    r2_score_b = lreg.r2_score(dataset.data, dataset.target.T, theta_b)
    #preise vorhersagen
    predicted_prize = lreg.LR_predict(dataset.data, theta_b)

    #vorhergesagt Preis über den eigentlichen Preis
    fig2 = plt.figure(figsize=(8,8))
    plt.scatter(predicted_prize,dataset.target.T)
    plt.xlabel('vorhergesagter Preis')
    plt.ylim(0,55)
    plt.ylabel('tatsächlicher Preis')
    plt.xlim(0,55)
    plt.plot(plt.xlim(), plt.ylim(),'--r')
    plt.title('Bestimmtheitsmaß {0:.5f}'.format(r2_score_b))
    plt.show()
