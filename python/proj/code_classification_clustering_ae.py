#!/usr/bin/env python
# coding: utf-8

#     Dependencies

# In[7]:


# Bibliotheken laden
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import time

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

import seaborn as sns

from scipy import stats
#import impute.SimpleImputer from sklearn
#from statsmodels.formula.api import ols

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score, GridSearchCV,RandomizedSearchCV, KFold
#from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import scorer, accuracy_score,confusion_matrix,classification_report, f1_score, recall_score, precision_score, accuracy_score, precision_recall_curve, roc_curve, roc_auc_score, auc, adjusted_rand_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC, OneClassSVM

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.cluster import KMeans, MiniBatchKMeans,DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn import tree
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

import joblib

#from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.under_sampling import RandomUnderSampler

from collections import Counter


# Hilfsfunktionen laden

import helper
get_ipython().run_line_magic('matplotlib', 'inline')


# Alle Algorithmen sollen den gleichen seed verwenden
seed = 42


#     **Data preperation**



time_start = time.time()
#Daten aus CSV in Pandas Dataframe einlesen
df=pd.read_csv('creditcard.csv', sep=',')   


#feature scaling
robust_scaler = RobustScaler()

df['ScaleAmount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Scaledtime'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1,1))
scaled_df = df.drop(['Time','Amount'],axis=1)
print(df.head())


df_classes = df['Class']

#drop label
scaled_df_no_Class = scaled_df.drop(['Class'],axis=1)


#drop irrelevant Features
scaled_df = scaled_df_no_Class.drop(['V9', 'V10','V11', 'V12', 'V13','V14','V15' ,'V16'],axis=1)

#Ein- und Ausgabewerte
x = scaled_df
y = df_classes

print("\nTrainingset contains missing values: ",x.isnull().values.any())

#Behandlung von fehlenden Werten
if x.isnull().values.any():
    mean = x.mean
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
    imputer = imputer.fit(x)
    x = imputer.fit_transform(x)


#aufteilen von Trainings und Testdaten (4:1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state = seed) 


print("\nxtrain.shape : ", xtrain.shape)
print("xtest.shape  : ", xtest.shape)
print("ytrain.shape : ", ytrain.shape)
print("ytest.shape  : ", ytest.shape)


# Balance überprüfen und ggf ausgleichen
nonfraud, fraud = df_classes.value_counts()
print("Imbalance of Dataset: Non-Fraud= ",nonfraud,"Fraud= ", fraud,"\n")

if nonfraud != fraud:
    # Resample the Data
    #undersampling
    rus = RandomUnderSampler(random_state=seed)
    #Generate the undersample data
    x_rus, y_rus = rus.fit_sample(xtrain, ytrain)
    x_test_rus, y_test_rus = rus.fit_sample(xtest,ytest)

    print("\nUndersampled data generated. New size: ",sorted(Counter(y_rus).items()),sorted(Counter(y_test_rus).items()))

    #SMOTE
    #Initialisieren
    os = SMOTE(random_state=0)
    #Generate the resample data
    os_res_x, os_res_y = os.fit_sample(xtrain,ytrain)
    os_test_x, os_test_y = os.fit_sample(xtest,ytest)

    #Counts of each class in oversampled data
    print("\nResampled data generated. New size: ",sorted(Counter(os_res_y).items()),sorted(Counter(os_test_y).items()))

    
#Datenset für semi-supervied learning in frauds und nichtfrauds aufteilen
xtrain_normal = xtrain[ytrain.values == 0]
xtrain_fraud = xtrain[ytrain.values == 1]

xtrain_rus_normal = x_rus[y_rus == 0]
xtrain_rus_fraud = x_rus[y_rus == 1]

xtest_normal = xtest[ytest.values == 0]
xtest_fraud = xtest[ytest.values == 1]

xtest_rus_normal = x_test_rus[y_test_rus == 0]
xtest_rus_fraud = x_test_rus[y_test_rus == 1]
    
    
    
#PCA
kpca = PCA(n_components=0.95) #95% Varianz beibehalten
os_res_x_kpca = kpca.fit_transform(os_res_x)
os_test_x_kpca = kpca.fit_transform(os_test_x)
cumsum = np.cumsum(kpca.explained_variance_ratio_)
print("\nExplained variance ratio after PCA: ",cumsum)  


df_pca = pd.DataFrame(os_res_x_kpca)



print('\nData preperation done! Time elapsed: {} seconds'.format(time.time()-time_start))


#     **Data Stats & Viz**

#stats
print(scaled_df.describe())

#ANOVA
model = ols('Class ~ ScaleAmount + V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V14 + V15 + V16 + V17 + V18 + V19 + V20 + V21 + V22 + V23 + V24 + V25 + V26 + V27 + V28 ', scaled_df).fit()
print(model.summary())
fresult = model.f_test(scaled_df)
print(fresult)

#plot sample distr.
with PdfPages('class_count.pdf') as pdf:
    fig = plt.figure(figsize=(8,6))
    count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
    ax = count_classes.plot(kind = 'bar')
    ax.set_xticklabels(['Not Fraud', 'Fraud'], rotation=0, ha='right')
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.xaxis.set_label_text('Class')
    ax.xaxis.label.set_visible(False)
    pdf.savefig(fig)
    
print("Sample ditr. done..")


with PdfPages('feature_distr.pdf') as pdf:
    #Verteilungen plotten
    v_features = scaled_df.columns
    fig = plt.figure(figsize=(12,31*4))
    gs = gridspec.GridSpec(31,1)
    

    for i, col in enumerate(v_features):
        ax = plt.subplot(gs[i])
        sns.distplot(scaled_df[col][df_classes.values==0],color='g',label='valide')
        sns.distplot(scaled_df[col][df_classes.values==1],color='b',label='fraud')
        ax.legend()
    pdf.savefig(fig)
    
print("deviations done..")


# In[ ]:


#correlation
with PdfPages('corr_heatmap.pdf') as pdf:
    fig = plt.figure(figsize=(8,6))
    ax = sns.heatmap(scaled_df[['V1','V2','V3','V4','V5','V6','V7','V8','V17','V18','V19','V20','Class','ScaleAmount','Scaledtime']].corr(),linewidths=0, vmin=-1, vmax=1, center=0)
    ax.set_xlabel("Feature A")
    ax.set_ylabel("Feature B")
    pdf.savefig(fig)
    
print("Heatmap done..")


# In[ ]:


#scatter matrix
with PdfPages('pairplot.pdf') as pdf:
    fig = plt.figure(figsize=(40,40))
    #sns.pairplot(scaled_df[['V1','V2','V3','V4','V5','V6','V7','V8','V17','V18','V19','V20','Class','ScaleAmount','Scaledtime']], hue='Class');
    sns.pairplot(scaled_df[['V1','V2','Class','ScaleAmount','Scaledtime']], hue='Class');
    pdf.savefig(fig)
    
print("Pairplot done..")


# In[ ]:


#Elbow Curve for identifying the best number of dimensions
expl_var= []
for k in range(1, 30):
    print('pca for: ',k)
    pca = PCA(n_components=k, random_state=seed)
    pca_reduced = pca.fit_transform(os_res_x)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    expl_var.append(np.mean(cumsum))

print("calculated. now plotting..")
print(expl_var)

with PdfPages('explained_var.pdf') as pdf:
    fig = plt.figure(figsize=(12, 6))
    plt.grid(linestyle=':', linewidth=0.5)
    plt.plot(range(1, 30), expl_var)
    plt.title('The Elbow Method')
    plt.xlabel('Number of dimensions - d')
    plt.ylabel('Explained Variance')
    pdf.savefig(fig)
    
    #pdf.close()
    plt.show()


#     **Data Viz TSNE/PCA/SVD**

#use undersampled data for runtime and visibility reasons
[X_reduced_tsne,X_reduced_pca] = helper.plot_dimensions(x_rus,y_rus)
#[X_os_tsne,X_os_pca] = helper.plot_dimensions(scaled_df_no_Class, df_classes)
#[xtrain_tsne,xtrain_pca]helper.plot_dimensions(xtrain,ytrain)

#print("old shape: ",x_rus.shape,"\nshape after t-sne: ",X_reduced_tsne.shape)


#     **Parameter tuning mittels Gridsearch**


time_start = time.time()

#parameter definieren
parameter_grid_logistic = {
     #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
     'intercept_scaling': [0.5,1,2],
     'C': [2,2.5,3],
     'max_iter': [1900,2300,2700],
     #'dual': [True, False],
     'solver': ['sag', 'saga', 'liblinear','lbfgs'],
    #'n_jobs':[-1],
    'random_state': [seed]
}
model_logistic = LogisticRegression()


parameter_grid_tree = {
    'max_depth': [4, 5, 6, 7], 
    'min_samples_split': [2,3],
    'max_features': ['sqrt', 'auto', 'log2', None],
    #'n_jobs':[-1],
    'random_state': [seed]
}
model_tree = tree.DecisionTreeClassifier()

parameter_grid_SVC = {
    'probability': [True],
    'kernel': ['linear', 'rbf', 'sigmoid','poly'],
    #'penalty': ['l2', 'l1'],
    'C': [26,30,32],
    'max_iter': [-1],
    'gamma': [0.0006, 0.00055,0.00065],
    'tol': [0.000001,0.0000005, 0.000005],
    #'n_jobs':[-1],
    'random_state': [seed]
}
model_SVC = SVC()


parameter_grid_SGD = {
    'alpha': [0.00025,0.0003,0.00035],
    'tol':[0.0030,0.0035,0.0040],
    'power_t': [0.5,0.4,0.6],
    'class_weight': [None],
    'epsilon':[0.0002,0.0005,0.001],
    'loss': ['modified_huber', 'squared_hinge'],# 'perceptron'],
    'penalty': ['l2', 'l1', None, 'elasticnet'],
    'max_iter': [600,700,800],
    #'n_jobs':[-1],
    'random_state': [seed]
}
model_SGD = SGDClassifier()


parameter_grid_knearest = {
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size':[3,5,8,12],
    'metric':['minkowski'],
    'metric_params':[None],
    'n_neighbors':[2,5,10],
    'p':[2,3,4],
    'weights': ['uniform', 'distance'],
    'n_jobs':[-1],
    #'random_state': [seed]
}
model_knearest = KNeighborsClassifier()


#Gridsearch with undersampled dataset
grid_search_logistic = GridSearchCV(model_logistic, parameter_grid_logistic, cv=6, n_jobs=-1).fit(x_rus,y_rus) #cv: plit dataset into k consecutive folds
grid_search_tree = GridSearchCV(model_tree, parameter_grid_tree, cv=6, n_jobs=-1).fit(x_rus,y_rus) #cv: plit dataset into k consecutive folds
grid_search_svc = GridSearchCV(model_SVC, parameter_grid_SVC, cv=6, n_jobs=-1).fit(x_rus,y_rus) #cv: plit dataset into k consecutive folds
grid_search_sgd = GridSearchCV(model_SGD, parameter_grid_SGD, cv=6, n_jobs=-1).fit(x_rus,y_rus) #cv: plit dataset into k consecutive folds
grid_search_knear = GridSearchCV(model_knearest, parameter_grid_knearest, cv=6, n_jobs=-1).fit(x_rus,y_rus) #cv: plit dataset into k consecutive folds


print('\nParameter estimation done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[14]:


#GS scores ausgeben

grid_search = [grid_search_logistic,grid_search_tree,grid_search_svc,grid_search_sgd,grid_search_knear]


for grids in grid_search:

    cvres = grids.cv_results_
    print("\n",type(grids.estimator).__name__,"highest score: ",grids.best_score_," with: ",grids.best_params_)



#print feature importance (nur für Decision tree relevant)

feature_importance = grid_search.best_estimator_.feature_importances_
feature_importance = f_importances(svm.coef_, scaled_df.columns.values)
for features in sorted(zip(feature_importance,scaled_df.columns.values),reverse=True)
   #print(features)


#==========================================================================================================================================================================
#
#     **Supervised Modelle**
#
# In[15]:


time_start = time.time()

#logistic regression
log_reg = LogisticRegression(**grid_search_logistic.best_params_)
#cv_train_logistic, cv_test_logistic = helper.build_model_train_test(log_reg, os_res_x, os_test_x, os_res_y, os_test_y,6)
cv_train_logistic, cv_test_logistic = helper.build_model_train_test(log_reg, x_rus,x_test_rus, y_rus, y_test_rus,6)


print('\nLogistic regression done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[16]:


time_start = time.time()

#decision tree
dt_classifier = tree.DecisionTreeClassifier(**grid_search_tree.best_params_)
cv_train_tree, cv_test_tree = helper.build_model_train_test(dt_classifier, x_rus,x_test_rus, y_rus, y_test_rus,6)
#cv_train_tree, cv_test_tree =helper.build_model_train_test(dt_classifier,os_res_x,os_test_x,os_res_y,os_test_y,6)


print('\nNow plotting tree....'.format(time.time()-time_start))

fig = plt.figure(figsize=(16,8))

tree.plot_tree(dt_classifier) 

print('\nDecision tree done! Time elapsed: {} seconds'.format(time.time()-time_start))


time_start = time.time()
#Linear Support Vector (Support Vector Machines (SVM))

sv_classifier = SVC(**grid_search_svc.best_params_)
cv_train_svc, cv_test_svc = helper.build_model_train_test(sv_classifier, x_rus,x_test_rus, y_rus, y_test_rus,6)
#cv_train_svc, cv_test_svc = helper.build_model_train_test(sv_classifier,os_res_x,os_test_x,os_res_y,os_test_y,6)

print('\nValidation done! Time elapsed: {} seconds. Now plotting Support Vectors.'.format(time.time()-time_start))

#plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#plot_svc_decision_function(model);


print('\nSupport Vector Classifier done! Time elapsed: {} seconds'.format(time.time()-time_start))



time_start = time.time()
#k-nearest with plot
knearest_classifier = KNeighborsClassifier(**grid_search_knear.best_params_)
#cv_train_knearest, cv_test_knearest = helper.build_model_train_test(knearest_classifier, x_rus,x_test_rus, y_rus, y_test_rus,6)
cv_train_knearest, cv_test_knearest = helper.build_model_train_test(knearest_classifier,os_res_x,os_test_x,os_res_y,os_test_y,6)

#print(len(os_res_x),len(os_res_y))
#knearestwithplot(15,os_res_x,os_res_y,knearest_classifier)
print('\nk- done! Time elapsed: {} seconds'.format(time.time()-time_start))




#Boxplot the AUC
#
results = np.array([["logistic regression",np.sort(cv_train_logistic), np.sort(cv_test_logistic)],
          ["decision tree",np.sort(cv_train_tree), np.sort(cv_test_tree)],
          ["svm",np.sort(cv_train_svc),np.sort( cv_test_svc)],
          #["sgd",np.sort(cv_train_sgd),np.sort(cv_test_sgd)],
          ["knearest",np.sort(cv_train_knearest), np.sort(cv_test_knearest)]])

#print(results)

#save results
joblib.dump(results,"cv_results.txt",protocol=3)

labels=results[0:5,0]
cv_auc = results[0:5,1]
#cv_testscores = results[0:5,2]

trainscores={}
for i in range(0, len(cv_auc)):
    trainscores[labels[i]]=cv_auc[i]

#testscores={}
#for i in range(0, len(cv_testscores)):
#    testscores[labels[i]]=cv_testscores[i]
    

df_trainscores = pd.DataFrame(trainscores)
#df_testscores = pd.DataFrame(testscores)

with PdfPages('cvalidation_score_traindata.pdf') as pdf:
    fig = plt.figure(figsize=(12,6))
    df_trainscores.boxplot();
    #gs = gridspec.GridSpec(5,1)

    plt.title("AUC on Testdata")
    plt.xlabel("Klassifizierer")
    plt.ylabel("CV Score")
    plt.ylim([0.942,0.988])
    plt.grid(linestyle=':', linewidth=.95)
    pdf.savefig(fig)
        
plt.show()



#     **estimate Cluster**


#Elbow Curve for identifying the best number of clusters
wcss = [] # Within Cluster Sum of Squares
X_reduced_pca = PCA(n_components=2, random_state=seed).fit_transform(xtrain)
for k in range(1, 21):
    kmeans = KMeans(n_clusters = k,init = 'k-means++',precompute_distances=True, random_state = 0, n_jobs=-1)
    kmeans.fit(x_rus)
    wcss.append(kmeans.inertia_)
print("now ploting..")
with PdfPages('kmeans_elbow_undersampled.pdf') as pdf:   
    fig = plt.figure(figsize=(12, 6))
    plt.plot(range(1, 21), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters - k')
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    plt.ylabel('WCSS')
    pdf.savefig(fig)
    plt.show()


#     **Silhouette Score**



#red line represents the avg. silhouette score
X_reduced_pca = PCA(n_components=2, random_state=seed).fit_transform(x_rus)
helper.plot_silhouette_scores(X_reduced_pca,ytrain,[2])



#==========================================================================================================================================================================
#
#     **Unsupervised Modelle**

# In[ ]:


#parameter definieren

parameter_grid_kmeans = {
    'n_clusters':[2],
    'random_state': [seed]
}
model_kmeans = KMeans()

parameter_grid_dbscan = {
    'eps':[25,30,35,40]
}
model_dbscan = DBSCAN()


#grid_search_kmeans = GridSearchCV(model_kmeans, parameter_grid_kmeans, cv=6, n_jobs=-1).fit(x_rus,y_rus) #cv: plit dataset into k consecutive folds
#grid_search_dbscan = GridSearchCV(model_dbscan, parameter_grid_dbscan, cv=6, n_jobs=-1, scoring=scorer.make_scorer(accuracy_score,greater_is_better=True)).fit(x_rus,y_rus) #cv: plit dataset into k consecutive folds


#grid_search = [grid_search_kmeans,model_dbscan]


#for grids in grid_search:

    #cvres = grids.cv_results_
    
    #print highest score
    #print("\n",type(grids.estimator).__name__,"highest score: ",grids.best_score_," with: ",grids.best_params_)


# In[47]:


X_reduced_tsne = TSNE(n_components=3, random_state=seed).fit_transform(x_rus)
X_reduced_pca = PCA(n_components=0.95).fit_transform(x_rus)


time_start = time.time()

# k-means - define clusters as distances around centroids
kmeans_best = KMeans(n_clusters = 2,init = 'k-means++',precompute_distances=True, random_state = seed, n_jobs=-1)
train_clusters = kmeans_best.fit(X_reduced_tsne)

#Perfect labeling is scored 1.0
helper.plot_Cluster(train_clusters,X_reduced_tsne,y_rus)

print('\nClustering done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[65]:


time_start = time.time()

# DBSCAN - define clusters as continuous regions of high densitiy
dbscan = DBSCAN(algorithm='auto', eps=2, leaf_size=20, metric='euclidean', metric_params=None, min_samples=1, n_jobs=-1, p=3)
train_db = dbscan.fit(X_reduced_tsne)
#train_db = dbscan.fit(x_rus)

helper.plot_Cluster(train_db,X_reduced_tsne,y_rus)


print('\nClustering done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[56]:


time_start = time.time()

#X_reduced_tsne = pd.read_csv('creditcard_tsne_2d.csv', sep=';')
print('csv read..')
#X_reduced_pca = PCA(n_components=0.95).fit_transform(x_rus)
print('pca done...')
# Gaussian Mixtures (Outlier detection)
bayesianGM = BayesianGaussianMixture(n_components=2, covariance_type='full', tol=0.0001, reg_covar=1e-06, max_iter=1000, n_init=1, init_params='kmeans', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None, random_state=seed, warm_start=False, verbose=0, verbose_interval=10)
# Fit a Dirichlet process Gaussian mixture using two components
train_bay = bayesianGM.fit(X_reduced_tsne)

helper.plot_Mixture(train_bay,X_reduced_tsne,y_rus)

print('\nClustering done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[46]:


time_start = time.time()
# Local Outlier Factor (Outlier detection


LOF_cluster = LocalOutlierFactor(n_neighbors=3, algorithm='auto', leaf_size=3, metric='minkowski', p=2, metric_params=None, contamination='legacy', novelty=True, n_jobs=-1)
train_LOF = LOF_cluster.fit(X_reduced_pca)

#Perfect labeling is scored 1.0
helper.plot_LOF(train_LOF,X_reduced_pca,y_rus,-1.5)

#print("Adjusted random score is: ",score)
print('\nClustering done! Time elapsed: {} seconds'.format(time.time()-time_start))



#     **Artificial Neuronal Networks (AAN)**

# In[18]:


# Autoencoder - Training on non-Frauds
# high reconstruction-error indicates frauds

X_reduced_pca = PCA(n_components=0.95, random_state=seed).fit_transform(os_res_x)
X_test_reduced_pca = PCA(n_components=0.95, random_state=seed).fit_transform(os_res_x)

#input_dim = os_res_x[os_res_y==0].shape[1]
input_dim = X_reduced_pca[os_res_y==0].shape[1]
encoding_dim = 8  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_layer = Input(shape=(input_dim, ))

# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

# "decoded" is the lossy reconstruction of the input
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

# this model maps an input to its reconstruction

#check if model already exists in HDD

my_file = Path("autoencoder3.h5")

if my_file.is_file():
    autoencoder = load_model("autoencoder3.h5")
else:
    autoencoder = Model(inputs=input_layer, outputs=decoder)


#training

nb_epoch = 100
batch_size = 8

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="autoencoder3.h5",
                               verbose=0,
                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_reduced_pca[os_res_y==0],X_reduced_pca[os_res_y==0],
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,                                                                 
                    validation_data=(X_test_reduced_pca, X_test_reduced_pca),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


# In[42]:


#evaluate AE
autoencoder = load_model("autoencoder3.h5")


with PdfPages('reconstruction_validation.pdf') as pdf:
#plot reconstruction error
    fig = plt.figure(figsize=(18,6))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Reconstruction error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right');
    plt.grid(linestyle=':', linewidth=0.5)
    pdf.savefig(fig)
    
plt.show()
    
with PdfPages('acurracy_ae.pdf') as pdf:
    fig = plt.figure(figsize=(18,6))
    plt.ylabel('error')
    plt.xlabel('data index')
    plt.ylim((0, 1000))
    plt.title('autoencoder reconstruction error')
    
    plt.grid(linestyle=':', linewidth=0.5)
    
    for i in [1,0]:
        predictions = autoencoder.predict(X_reduced_pca[os_res_y==i])
        mse = np.mean(np.power(X_reduced_pca[os_res_y==i] - predictions, 2), axis=1)

        error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': os_res_y[os_res_y==i]})
        error_df.sort_index(inplace=True)
        plt.plot(error_df['reconstruction_error'])
    
    plt.legend(['Fraud','non Fraud'], loc='upper right');
    pdf.savefig(fig)

plt.show()


#     **re-use pretrained data**

# In[ ]:


os_res_x[os_res_y==0].shape[1]


# In[ ]:




