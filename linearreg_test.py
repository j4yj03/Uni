import numpy as np
import linearreg as lreg



if __name__ == "__main__":

    x = np.matrix([[1, 2], [4, 5]])
    print(x,x.shape)
    x1 = lreg.extend_matrix(x)

    print(x1,x1.shape)
