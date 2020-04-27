import numpy as np
import linearreg as lreg



if __name__ == "__main__":

    x = np.matrix([[1, 2, 4], [4, 5, 7],[5, 8, 9],[7,8,9]])
    print(x,x.shape)
    x1 = lreg.extend_matrix(x)

    print(x1,x1.shape)

    x2 = lreg.extend_matrix(x1)

    print(x2,x2.shape)
