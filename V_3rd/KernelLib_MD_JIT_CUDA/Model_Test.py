import numba as nb
import numpy as np
import matplotlib.pyplot as mpl
################################################################################
@nb.jit(nopython=True,nogil=True)
def A(x,y):
    #
    z=x+y
    return(z)
################################################################################
@nb.jit(nopython=True,nogil=True)
def B(x,y):
    #
    z=x-y
    return(z)
################################################################################
@nb.jit(nopython=True,nogil=True)
def C(model,x,y):
    #
    z=model(x,y)
    return(z)
################################################################################
def Criterion_Plot(Name,x,f1,f2,Y1L,Y1H,Y2L,Y2H):
    xl=int(x[0])
    xh=int(x[-1])
    y1l=Y1L
    y1h=min(Y1H,1.25*max(f1)-0.25*y1l)
    y2l=Y2L
    y2h=min(Y2H,1.25*max(f2)-0.25*y2l)
    #
    fig,ax1=mpl.subplots()
    lab1='UnAdsorbed-IntTime f Distribution'
    ax1.set_xlabel('$'+str(Name)+'$')
    ax1.set_xlim(xl,xh)
    ax1.set_ylabel('$f$', color='r')
    ax1.set_ylim(y1l,y1h)
    l1=ax1.plot(x,f1,'ro',markersize=4,label=lab1)
    #ax1.legend(loc='upper right',fontsize='x-small')
    ax1.tick_params(axis='y', labelcolor='r')
    #
    ax2=ax1.twinx()
    lab2='UnAdsorbed-IntTime CountN Distribution'
    ax2.set_ylabel('$CountN$', color='b')
    ax2.set_ylim(y2l,y2h)
    l2=ax2.plot(x,f2,'b--',markersize=4,label=lab2)
    ax2.tick_params(axis='y', labelcolor='b')
    #
    ls=l1+l2
    legs=[l.get_label() for l in ls]
    ax2.legend(ls,legs,loc='upper right',fontsize='x-small')
    #
    #fig=mpl.figure()
    #lab='UnAdsorbed-IntTime Distribution'
    #mpl.plot(x[j,:],f[j,:],'ro',markersize=4,label=lab)
    #mpl.legend(loc='upper right',fontsize='x-small')
    #mpl.xlabel('$'+str(Name)+'$')
    #mpl.ylabel('$f$')
    #mpl.axis([xl,xh,yl,yh])
    mpl.savefig('Criterion_'+Name+'.png',dpi=600)
    #
    #mpl.show()
    mpl.close()
################################################################################
def main():
    #
    x=1
    y=2
    z=C(A,x,y)
    X=np.arange(0.01, 10.0, 0.01)
    Y1=np.arange(0.01, 10.0, 0.01)
    Y2=np.arange(0.02, 20.0, 0.02)
    Criterion_Plot('X',X,Y1,Y2,0.0,20.0,0.0,40.0)
    print(z)
################################################################################
if __name__ == '__main__':
    #
    main()
