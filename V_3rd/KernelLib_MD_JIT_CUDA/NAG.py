import numpy as np
import numba as nb
import matplotlib.pyplot as mpl
################################################################################
def NAG_Method(v,x,mu,lr,dx):
    #
    pre_v=v
    v=mu*v
    v+=-lr*dx
    x+=v+mu*(v-pre_v)
    #
    return(v,x)
################################################################################
def Momentum_Method(v,x,mu,lr,dx):
    #
    v=mu*v
    v+=-lr*dx
    x+=v
    #
    return(v,x)
################################################################################
def Adam_Method(t,v,cache,x,mu,decay_rate,lr,dx):
    #
    eps=0.0
    #eps=1e-8
    v=mu*v+(1-mu)*dx
    vt=v/(1-mu**t)
    cache=decay_rate*cache+(1-decay_rate)*dx**2
    cachet=cache/(1-decay_rate**t)
    x+=-(lr/(np.sqrt(cachet)+eps))*vt
    #
    return(v,cache,x)
################################################################################
def GD_Method(x,lr,dx):
    #
    x+=-lr*dx
    #
    return(x)
################################################################################
def F(x):
    #
    y=np.cos(x)+2*np.cos(x/2)+3*np.cos(x/3)+5*np.cos(x/5)
    #y=3*x**4-4*x**3-12*x**2+17
    #y=np.sin(x)-np.cos(x)
    #
    return(y)
################################################################################
def Gradient(x,delta):
    #
    y1=F(x-delta)
    y2=F(x+delta)
    dx=(y2-y1)/(2*delta)
    #
    return(dx)
################################################################################
def Data():
    #
    mu=0.9995
    decay_rate=0.9995
    lr=1e-4
    Alr=1e-3
    ilr=1e-3
    delta=1e-4
    #
    v=0.0
    Mv=v
    Av=v
    cache=0.0
    x=-36.0
    Mx=x
    Ax=x
    xx=x
    ix=x
    lx=36
    N=20000
    #
    NAG=np.zeros((N+1,3))
    M=np.zeros((N+1,3))
    Adam=np.zeros((N+1,3))
    GD=np.zeros((N+1,3))
    THE=np.zeros((2*N+1,2))
    NAG[0,0]=x
    NAG[0,1]=F(x)
    NAG[0,2]=0
    M[0,0]=Mx
    M[0,1]=F(Mx)
    M[0,2]=0
    Adam[0,0]=Ax
    Adam[0,1]=F(Ax)
    Adam[0,2]=0
    GD[0,0]=xx
    GD[0,1]=F(xx)
    GD[0,2]=0
    THE[N,0]=ix
    THE[N,1]=F(ix)
    #
    for i in range(N):
        dx=Gradient(x,delta)
        v,x=NAG_Method(v,x,mu,lr,dx)
        Mdx=Gradient(Mx,delta)
        Mv,Mx=Momentum_Method(Mv,Mx,mu,lr,Mdx)
        Adx=Gradient(Ax,delta)
        Av,cache,Ax=Adam_Method(i+1,Av,cache,Ax,mu,decay_rate,Alr,Adx)
        dxx=Gradient(xx,delta)
        xx=GD_Method(xx,ilr,dxx)
        NAG[i+1,0]=x
        NAG[i+1,1]=F(x)
        NAG[i+1,2]=i+1
        M[i+1,0]=Mx
        M[i+1,1]=F(Mx)
        M[i+1,2]=i+1
        Adam[i+1,0]=Ax
        Adam[i+1,1]=F(Ax)
        Adam[i+1,2]=i+1
        GD[i+1,0]=xx
        GD[i+1,1]=F(xx)
        GD[i+1,2]=i+1
        THE[N-i-1,0]=ix-(i+1)/N*lx/5
        THE[N-i-1,1]=F(THE[N-i-1,0])
        THE[N+i+1,0]=ix+(i+1)/N*lx
        THE[N+i+1,1]=F(THE[N+i+1,0])
    #
    return(NAG,M,Adam,GD,THE)
################################################################################
def main():
    #
    NAG,M,Adam,GD,THE=Data()
    #
    fig1=mpl.figure()
    mpl.scatter(NAG[:,0],NAG[:,1],marker='o',color='',edgecolors='r',s=10,label='NAG')
    mpl.scatter(M[:,0],M[:,1],marker='^',color='',edgecolors='g',s=10,label='Momentum')
    mpl.scatter(Adam[:,0],Adam[:,1],marker='X',color='',edgecolors='b',s=10,label='Adam')
    mpl.scatter(GD[:,0],GD[:,1],marker='*',color='',edgecolors='m',s=10,label='GD')
    mpl.plot(THE[:,0],THE[:,1],'k--',markersize=4,label='THE')
    mpl.legend(loc='upper right',fontsize='xx-small')
    mpl.xlabel('$x$')
    mpl.ylabel('$y$')
    mpl.savefig('Function.png',dpi=600)
    #
    fig2=mpl.figure()
    mpl.scatter(NAG[:,2],NAG[:,0],marker='o',color='',edgecolors='r',s=10,label='NAG')
    mpl.scatter(M[:,2],M[:,0],marker='^',color='',edgecolors='g',s=10,label='Momentum')
    mpl.scatter(Adam[:,2],Adam[:,0],marker='X',color='',edgecolors='b',s=10,label='Adam')
    mpl.scatter(GD[:,2],GD[:,0],marker='*',color='',edgecolors='m',s=10,label='GD')
    mpl.legend(loc='upper right',fontsize='xx-small')
    mpl.xlabel('$Loops$')
    mpl.ylabel('$x$')
    mpl.savefig('x_Loop.png',dpi=600)
    #
    fig3=mpl.figure()
    mpl.scatter(NAG[:,2],NAG[:,1],marker='o',color='',edgecolors='r',s=10,label='NAG')
    mpl.scatter(M[:,2],M[:,1],marker='^',color='',edgecolors='g',s=10,label='Momentum')
    mpl.scatter(Adam[:,2],Adam[:,1],marker='X',color='',edgecolors='b',s=10,label='Adam')
    mpl.scatter(GD[:,2],GD[:,1],marker='*',color='',edgecolors='m',s=10,label='GD')
    mpl.legend(loc='upper right',fontsize='xx-small')
    mpl.xlabel('$Loops$')
    mpl.ylabel('$y$')
    mpl.savefig('y_Loop.png',dpi=600)
    #
    #mpl.show()
    mpl.close()
################################################################################
if __name__ == "__main__":
    main()
