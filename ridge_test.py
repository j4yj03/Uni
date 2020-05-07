import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import ridge_regression as ridge

from itertools import chain, combinations_with_replacement

def columnNames(listToExtend, degree):
    list = listToExtend
    label = ""
    combinations =  chain.from_iterable(combinations_with_replacement(listToExtend,i) for i in range(degree, degree+1))
    for combi in combinations:
        print(combi)
        for i in range(degree):
            print(combi[degree-1])
            label=label+str(combi[degree-1])

        list.append(label)
        label = ""

    print(label)

    return list



if __name__ == "__main__":

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    degree = 2


    multi = pd.read_csv("./Uni/data/multivariat.csv", sep=',')
    print(multi.head())
    x1_m = multi['x1'].to_numpy()
    x2_m = multi['x2'].to_numpy()
    #print(multi[:20])

    X_m = multi.iloc[:,0:3].to_numpy()

    y_m = multi['y'].to_numpy()

    lab = columnNames(list(multi.columns),degree)

    print(X_m[:20].shape)

    X_m2 = ridge.QuadraticFeatures_fit_transform(X_m[:20],degree)

    df = pd.DataFrame(X_m2, columns=lab)

    print(df.head())
