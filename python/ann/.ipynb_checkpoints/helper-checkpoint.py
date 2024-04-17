

def plotactivationfunctions():
    
    from scipy.special import expit, logit
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.arange(-5, 5, .1)

    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    #step
    ax1.step([-5,5],[-1,1], where='mid')
    #tanh
    ax1.plot(x, np.tanh(x))
    #arctan
    ax1.plot(x, np.arctan(x))
    #sigmoid
    ax1.plot(x, expit(x))
    #ReLU
    ax1.plot(x, np.maximum(0, x))
    #leakyReLU
    ax1.plot(x, np.maximum(0.01*x, x), '-.')
    ax1.set_ylim([-1.1, 1.1])
    #ax1.set_xlim([-1.1, 1.1])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(linestyle=':', linewidth=0.5)
    ax1.legend(['heavyside','tanh','actan','sigmoid','ReLU','leakyReLU'])
    ax1.title.set_text('Aktivierungsfunktionen')


    #step
    ax2.stem([0],[0.1], '-.')
    #tanh
    ax2.plot(x[:-1], np.diff(np.tanh(x)))
    #arctan
    ax2.plot(x[:-1], np.diff(np.arctan(x)))
    #sigmoid
    ax2.plot(x[:-1], np.diff(1 / (1 + np.exp(-x))))
    #ReLU
    ax2.plot(x[:-1], np.diff(np.maximum(0, x)))
    #ax2.set_ylim([-1.1, 1.1])
    ax2.set_xlim([-5, 5])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(linestyle=':', linewidth=0.5)
    ax2.title.set_text('Ableitung')

    plt.show()