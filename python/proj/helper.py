#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#
import numpy as np
import pandas as pd
import time
import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

import seaborn as sns

from scipy import stats, linalg
#import impute.SimpleImputer from sklearn
#from statsmodels.formula.api import ols

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score, GridSearchCV,RandomizedSearchCV, KFold, StratifiedKFold
#from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model

from joblib import Parallel, delayed
#from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.under_sampling import RandomUnderSampler

from collections import Counter


seed = 42

# In[ ]:


################################################################################################################################
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


# In[ ]:


################################################################################################################################
def build_model_train_test(model, x_train, x_test, y_train, y_test, cv_splits): 
    print("\nDatashape {}".format(x_train.shape))
    
    y_train_score = model.fit(x_train,y_train)
    
    print("\nModel fit! Predict scores on trainingdata...")
    y_pred = model.predict(x_train)

    
    print("\n----------Accuracy Scores on Train data------------------------------------")
    print("F1 Score: ", metrics.f1_score(y_train,y_pred))
    print("Precision Score: ", metrics.precision_score(y_train,y_pred))
    print("Recall Score: ", metrics.recall_score(y_train,y_pred))
   # print("Cross Validation mean score: ",  cv_train_scores.mean()," (Std.: ", cv_train_scores.std(),")")

    print("\nModel fit! Predict scores on testdata...")
    y_pred_test = model.predict(x_test)
    #probs_test = model.predict_proba(x_test)
    
    #cv_test_scores= cross_val_score(model, x_test,y_test, cv=cv_splits)
    
    print("\n----------Accuracy Scores on Test data------------------------------------")
    print("F1 Score: ", metrics.f1_score(y_test,y_pred_test))
    print("Precision Score: ", metrics.precision_score(y_test,y_pred_test))
    print("Recall Score: ", metrics.recall_score(y_test,y_pred_test))
   # print("Cross Validation mean score: ",  cv_test_scores.mean()," (Std.: ", cv_test_scores.std(),")")

    cv = StratifiedKFold(n_splits=cv_splits)
    
        
    with PdfPages('validation train {}.pdf'.format(type(model).__name__)) as pdf:
        #Confusion Matrix Trainingdata
        fig1 = plt.figure(figsize=(18,6))
        gs = gridspec.GridSpec(1,2)

        ax1 = plt.subplot(gs[0])
        cnf_matrix = metrics.confusion_matrix(y_train,y_pred)
        row_sum = cnf_matrix.sum(axis=1,keepdims=True)
        #print(cnf_matrix)
        #print(row_sum)
        cnf_matrix_norm =cnf_matrix / row_sum
        sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True, xticklabels=["non-Fraud","Fraud"], yticklabels=["non-Fraud","Fraud"])
        #sns.
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        plt.title("Normalized Confusion Matrix - Train Data")


        #ROC Curve Trainingdata 
        tprs1 = []
        aucs1 = []
        mean_fpr = np.linspace(0, 1, 100)

    #===========================================================================================================
        ax3 = plt.subplot(gs[1])
        i = 1
        print("\nROC Curve for trainingdata...")
        for train, test in cv.split(x_train, y_train):
            probas_ = model.predict_proba(x_train[test])
            #probas_ = model.fit(x_train[train], y_train[train]).predict_proba(x_train[test])

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = metrics.roc_curve(y_train[test], probas_[:, 1])
            tprs1.append(np.interp(mean_fpr, fpr, tpr))
            tprs1[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs1.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
            
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs1, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs1)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs1, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve -  Traindata')
        plt.legend(loc="lower right")
        
        pdf.savefig(fig1)
        #pdf.close()
        
    with PdfPages('validation test {}.pdf'.format(type(model).__name__)) as pdf:
        fig2 = plt.figure(figsize=(18,6))
     #==============================================================================================================
        #Confusion Matrix Testdata
        gs1 = gridspec.GridSpec(1,2)
        ax2 = plt.subplot(gs1[0])
        cnf_matrix = metrics.confusion_matrix(y_test,y_pred_test)
        row_sum = cnf_matrix.sum(axis=1,keepdims=True)
        cnf_matrix_norm =cnf_matrix / row_sum
        sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True, xticklabels=["non-Fraud","Fraud"], yticklabels=["non-Fraud","Fraud"])
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        plt.title("Normalized Confusion Matrix - Test Data")


        #ROC Curve Testdata
        tprs2 = []
        aucs2 = []
        mean_fpr = np.linspace(0, 1, 100)

    #===========================================================================================================
        ax4 = plt.subplot(gs1[1])
        i = 1
        print("\nROC Curve for testdata...")
        for train, test in cv.split(x_test, y_test):
            probas_ = model.predict_proba(x_test[test])
            #probas_ = model.fit(x_test[train], y_test[train]).predict_proba(x_test[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = metrics.roc_curve(y_test[test], probas_[:, 1])
            tprs2.append(np.interp(mean_fpr, fpr, tpr))
            tprs2[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs2.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs2, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs2)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs2, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve -  Testdata')
        plt.legend(loc="lower right")

     #==============================================================================================================
        pdf.savefig(fig2)
        
    plt.show

    return [aucs2,tprs2]

# In[ ]:


###############################################################################################################################    
def knearestwithplot(n_neighbors,X,y,clf):
    h = .02  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        #clf = KNeighborsClassifier(n_neighbors, weights=weights)
        #clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        print(len(xx),len(yy))
        print(len(xx.ravel()),len(yy.ravel()))
        print(xx.shape, yy.shape, np.c_[xx.ravel(), yy.ravel()].shape)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

    plt.show()


# In[ ]:


##############################################################################################################################
def plot_svc_decision_function(model, ax=None, plot_support=True):
#Plot the decision function for a two-dimensional SVC
    if ax is None:
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        
        Y, X = np.meshgrid(y, x)
        
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)
        # plot decision boundary and margins
        ax.contour(X, Y, P, colors='k',
        levels=[-1, 0, 1], alpha=0.5,
        linestyles=['--', '-', '--'])
        # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=300, linewidth=1, facecolors='none');
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


# In[ ]:


############################################################################################################################
def plot_decision_tree(X, y, clf):
    # Parameters
    n_classes = 2
    plot_colors = "ryb"
    plot_step = 0.02
    
    for pairidx, pair in enumerate([[0, 1], [0, 2],
                                    [1, 2], [1, 1]]):
        # We only take the two corresponding features
        #X = iris.data[:, pair]
        #y = iris.target

        # Train
        #clf = DecisionTreeClassifier().fit(X, y)

        # Plot the decision boundary
        plt.subplot(2, 2, pairidx + 1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        print(np.c_[xx.ravel(), yy.ravel()].shape)
        #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = clf.predict(X)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        #plt.xlabel(iris.feature_names[pair[0]])
        #plt.ylabel(iris.feature_names[pair[1]])

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")

    plt.figure()
    clf = DecisionTreeClassifier().fit(iris.data, iris.target)
    plot_tree(clf, filled=True)
    plt.show()


# In[ ]:


##############################################################################################################################
def plot_dimensions(x,y):
    # T-SNE Implementation
    t0 = time.time()
    X_reduced_tsne = TSNE(n_components=2, random_state=seed).fit_transform(x)
    t1 = time.time()
    print("T-SNE took {:.2} s".format(t1 - t0))    

    # PCA Implementation
    t0 = time.time()
    X_reduced_pca = PCA(n_components=2, random_state=seed).fit_transform(x)
    t1 = time.time()
    print("PCA took {:.2} s".format(t1 - t0))

    # TruncatedSVD
    #t0 = time.time()
    #X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=seed).fit_transform(x)
    #t1 = time.time()
    #print("Truncated SVD took {:.2} s".format(t1 - t0))
    
    with PdfPages('dim_reduced.pdf') as pdf:
    
        fig = plt.figure(figsize=(32,12))
        #(ax1, ax2, ax3) = fig.subplots(1, 3)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        #ax3 = fig.add_subplot(133)
        # labels = ['No Fraud', 'Fraud']
        fig.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

        blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
        red_patch = mpatches.Patch(color='#AF0000', label='Fraud')




        # t-SNE scatter plot
        ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], s=4, c=(y == 0),
                    cmap='coolwarm', label='No Fraud', linewidths=2)
        ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], s=4, c=(y == 1),
                    cmap='coolwarm', label='Fraud', linewidths=2)
        ax1.set_title('t-SNE', fontsize=14)
        ax1.grid(True)
        ax1.legend(handles=[blue_patch, red_patch])

        # PCA scatter plot
        ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], s=4, c=(y == 0),
                    cmap='coolwarm', label='No Fraud', linewidths=2)
        ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], s=4, c=(y == 1),
                    cmap='coolwarm', label='Fraud', linewidths=2)
        ax2.set_title('PCA', fontsize=14)
        ax2.grid(True)
        ax2.legend(handles=[blue_patch, red_patch])

        # TruncatedSVD scatter plot
        #ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], s=4, c=(y == 0),
        #            cmap='coolwarm', label='No Fraud', linewidths=2)
        #ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], s=4, c=(y == 1),
        #            cmap='coolwarm', label='Fraud', linewidths=2)
        #ax3.set_title('Truncated SVD', fontsize=14)
        #ax3.grid(True)
        #ax3.legend(handles=[blue_patch, red_patch])

        pdf.savefig(fig)
        plt.show()


    return [X_reduced_tsne,X_reduced_pca]#,X_reduced_svd]
    
    
################################
def plot_silhouette_scores(X,y,n_range):
    range_n_clusters = n_range
    with PdfPages('kmeans_silhouette{}.pdf'.format(X.shape)) as pdf:  
        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 6)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 42 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters,init = 'k-means++',precompute_distances=True, random_state=seed, n_jobs=-1)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = metrics.silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
            pdf.savefig()
    plt.show()
###########################################################################################################################################
def plot_Cluster(model,X,y):
    print(model)
    # #############################################################################
    db = model
    labels_true = y
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    if (type(model).__name__ == "DBSCAN"):
        core_samples_mask[db.core_sample_indices_] = True
        
    labels = db.labels_
    #labels = y
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_) #,"Labels: ",np.bincount(labels)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Accuracy: %0.3f" % metrics.accuracy_score(labels_true, labels))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels,
                                               average_method='arithmetic'))
    #print("Silhouette Coefficient: %0.3f"
         # % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
    print("\nnow plotting...\n")
          
    #Confusion Matrix Testdata
    plt.figure(figsize=(18,6))
    gs1 = gridspec.GridSpec(1,2)
    ax2 = plt.subplot(gs1[0])
    cnf_matrix = metrics.confusion_matrix(labels,labels_true)
    row_sum = cnf_matrix.sum(axis=1,keepdims=True)
    cnf_matrix_norm =cnf_matrix / row_sum
    sns.heatmap(cnf_matrix_norm,cmap='YlGnBu',annot=True, xticklabels=["non-Fraud","Fraud"], yticklabels=["non-Fraud","Fraud"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.title("Normalized Confusion Matrix")      
    plt.show()      

    with PdfPages('cluster_kmean.pdf') as pdf:
        # Black removed and is used for noise instead.
        plt.figure(figsize=(24,12))
        plt.xticks(())
        plt.yticks(())
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        pdf.savefig()
        
    plt.show()
###########################################################################################################################################
    
def plot_Mixture(model,X,y):
    print(model)
    color_iter = itertools.cycle(['cornflowerblue','darkorange'])
    Y_= model.predict(X)
    probs = model.predict_proba(X)
    
    
    dpgmm = model
    means = dpgmm.means_
    #sigma = dpgmm.sigma
    covariances = dpgmm.covariances_
    index=1
    
    print("Accuracy: %0.3f" % metrics.accuracy_score(Y_, y))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(Y_, y))
    print("Completeness: %0.3f" % metrics.completeness_score(Y_, y))
    print("V-measure: %0.3f" % metrics.v_measure_score(Y_, y))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(Y_, y))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(Y_, y, average_method='arithmetic'))
    
    
    splot = plt.figure(figsize=(18,6))
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title('Bayesian Gaussian Mixture with a Dirichlet process prior')
    
########################################################################################################################################

def plot_LOF(model, X, y, thresh):
    # fit the model for outlier detection (default)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).

    ground_truth = y
    
    y_pred= model.predict(X)
    n_errors = (y_pred != ground_truth).sum()
    X_scores = model.negative_outlier_factor_
    
    X_norm = X[X_scores > thresh]
    
    X_outl = X[X_scores <= thresh]
    
    fig = plt.figure(figsize=(16,8))
    plt.title("Local Outlier Factor (LOF)")
    
    plt.scatter(X[:, 0], X[:, 1], color='k', s=1., label='Data points')

    # plot normalized circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())


    print("Outliers :",len(X_outl))
    print("Highest negative outlier factor: ",X[X_scores==min(X_scores)]," with ",min(X_scores))

    plt.scatter(X_norm[:, 0], X_norm[:, 1], s=10, edgecolors='green', facecolors='none', label='kein Outlier')
    plt.scatter(X_outl[:, 0], X_outl[:, 1], s=1000 * radius[X_scores <= thresh], edgecolors='red', facecolors='none', label='outlier score')

    plt.axis('tight')
    #plt.xlim((-4, 8))
    #plt.ylim((-4, 16))
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("found %d outliers with tresh = %d. Estimated %d wrong." % (len(X_outl),thresh,n_errors))
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]

    plt.show()
#########################################################################################################################################
    
def plot_OCSVM(model, X_train,X_test,y_train,y_test):
    # fit the model
    clf = model
    
    
    xx, yy = np.meshgrid(np.linspace(-100, 100, 500), np.linspace(-100, 100, 500))
    
    
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    #print(y_pred_train)
    #print(y_pred_test)
    y_pred_train_normal = y_pred_train[y_pred_train == 1]
    y_pred_train_outliers = y_pred_train[y_pred_train == -1]
    y_pred_test_normal = y_pred_test[y_pred_test == 1]
    y_pred_test_outliers = y_pred_test[y_pred_test == -1]
    
    
    #y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    #n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    
    print("\nScores on Traindata: ")
    print("Accuracy: %0.3f" % metrics.accuracy_score(y_pred_train, y_train))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_pred_train, y_train))
    print("Completeness: %0.3f" % metrics.completeness_score(y_pred_train,y_train))
    print("V-measure: %0.3f" % metrics.v_measure_score(y_pred_train, y_train))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y_pred_train, y_train))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(y_pred_train, y_train, average_method='arithmetic'))
    
    
    print("\nScores on Testdata: ")
    print("Accuracy: %0.3f" % metrics.accuracy_score( y_pred_test, y_test))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score( y_pred_test, y_test))
    print("Completeness: %0.3f" % metrics.completeness_score( y_pred_test, y_test))
    print("V-measure: %0.3f" % metrics.v_measure_score( y_pred_test, y_test))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score( y_pred_test, y_test))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score( y_pred_test, y_test, average_method='arithmetic'))
    
    
    
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(24,18))
    plt.title("Outlier Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 30
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[y_pred_test == 1][:, 0], X_test[y_pred_test == 1][:, 1], c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(X_test[y_pred_test == -1][:, 0], X_test[y_pred_test == -1][:, 1], c='gold', s=s, edgecolors='k')
    plt.axis('tight')
    plt.xlim((-40, 40))
    plt.ylim((-40, 40))
    plt.legend([a.collections[0], b1, b2,c],
               ["trained boundaries", "trained transactions",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d ; errors novel regular: %d ; "
        % (n_error_train, n_error_test))
    plt.show()
    
    
def plot_AE(mode,X,y):
    return 0