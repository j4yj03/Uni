import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec

#import python.logit.logistische_regression as lr
import logistische_regression as lr

from sklearn.datasets import load_iris


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# In[92]:

dataset = load_iris()

print(dataset.DESCR)

X = dataset.data[0:100]
y = dataset.target[0:100]
#print(X,y)

#print(X)

features = dataset.feature_names
target = dataset.target_names
#print(target)
#df = pd.DataFrame(data=X, columns=y)

mask = np.ones(np.size(dataset.data,1), dtype=bool)
mask[[1,3]] = False
X = X[:,mask]
#features = features[:mask]

# X_1=X[:,0]
# X_2=X[:,2]
# X = np.zeros(shape=(dataset.data.shape[0],2)
#
# X[:,0] = X_1
# X[:,1] = X_2

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30,  random_state=0)


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


theta, J = lr.LogisticRegression_fit(X_train, y_train, .1, 3e-5)

print(theta,J)


y_pred = lr.LogisticRegression_predict(X_test, theta)

acc = lr.accuracy_score(y_test,y_pred)
precision = lr.precision_score(y_test,y_pred)
#precision2 = lr.precision_score2(y_test,y_pred)
recall = lr.recall_score(y_test,y_pred)
#recall2 = lr.recall_score2(y_test,y_pred)

print('Accuracy: ',acc,'\nPrecision: ',precision,'\nRecall: ', recall)

#print('sk_acc', accuracy_score(y_test, y_pred),'\nsk_pre', precision_score(y_test, y_pred),'\nsk_rec', recall_score(y_test, y_pred))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .02  # step size in the mesh

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = lr.LogisticRegression_predict(np.c_[xx.ravel(), yy.ravel()], theta)
Z = Z.reshape(xx.shape)


x_values = [np.min(X_test[:, 0]) - 1, np.max(X_test[:, 0])+1]
y_values = - (theta[0] + np.dot(theta[1], x_values)) / theta[2]

#canvas
fig1 = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2)
#subplots
ax = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
#plot
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_title("trainset")
ax.scatter(X_train[:,0],X_train[:,1], c=y_train, edgecolors='black')
ax.legend(target)
ax1.set_xlabel(features[0])
ax1.set_ylabel(features[1])
ax1.set_title("testdata and decision function")
ax1.plot(x_values,y_values, color='black')
ax1.pcolormesh(xx, yy, Z, alpha=0.2)
ax1.scatter(X_test[:,0],X_test[:,1], c=y_test, edgecolors='black')
ax1.legend(target)
plt.show()
