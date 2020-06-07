import numba as nb
import numpy as np
import math
import matplotlib.pyplot as mpl
################################################################################
@nb.jit(nopython=True,nogil=True)
def I0(I0In):
    N_theta=1000
    d_theta=np.pi/N_theta
    I0=0.0
    for i in range(N_theta):
        I0+=math.exp(I0In*math.cos(d_theta*(i+0.5)))
    I0*=d_theta/np.pi
    return(I0)
################################################################################
@nb.jit(nopython=True,nogil=True)
def ttt(tt):
    ui=3.0
    ur=3.0
    I0In=2*(1-tt)*ur*ui/(tt*(2-tt))
    return(I0In)
################################################################################
def main():
    #X=[i/1000-1.0 for i in range(2001)]
    #Y=[I0(x) for x in X]
    X=[i/1000 for i in range(1,2000)]
    Y=[ttt(x) for x in X]
    mpl.plot(X,Y)
    mpl.show()
################################################################################
if __name__ == '__main__':
    main()
