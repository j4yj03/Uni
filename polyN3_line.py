#!/usr/local/bin/python3

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# some 3-dim points
n=20
xx=np.linspace(-12,5,num=n)
yy=np.linspace(-20,20,num=n)*2.4
zz=np.arange(n)**3

data = np.c_[xx,yy,zz]


# Points at which we will evaluate the line.
doInterp=True
if doInterp:
    print('Interpolating')
    evalN=20
    from scipy.interpolate import interp1d
    f = interp1d(xx, yy)
    XX = np.linspace(min(xx),max(xx),num=evalN);

    YY = f(XX)
else:
    XX=xx
    YY=yy


order = 3  # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[np.ones(data.shape[0]), xx, yy]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    print(C)
    Z = C[0] + C[1]*XX + C[2]*YY

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    print(C)
    Z = C[0] + C[1]*XX + C[2]*YY + C[3]*XX*YY + C[4]*XX**2 + C[5]*YY**2


elif order == 3:
    # best-fit cubic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2, data[:,:2]**3]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    print(C)
    Z = C[0] + C[1]*XX + C[2]*YY + C[3]*XX*YY + C[4]*XX**2 + C[5]*YY**2 + C[6]*XX**3 + C[7]*YY**3



# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.plot(XX, YY, Z)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
#ax.axis('equal')
#ax.axis('tight')
plt.show()
