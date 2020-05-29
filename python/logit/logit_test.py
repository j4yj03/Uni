import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import python.logit.logistische_regression as lr

from sklearn.datasets import load_iris



# In[92]:

dataset = load_iris()

print(dataset.DESCR)

X = dataset.data
y = dataset.target

features = dataset.feature_names
target = dataset.target_names

print(y[:50])

acc = lr.accuracy_score(y[:50],y[50:100])
print(acc)
