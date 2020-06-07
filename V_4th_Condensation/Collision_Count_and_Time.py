import numpy as np
import numba as nb
import math
import matplotlib.pyplot as mpl
import matplotlib.ticker as ticker
################################################################################
def ReadFile(FileName):
    #
    with open(FileName,'r') as In:
        Data=In.readlines()
        InData=Data[1:]
        ID=[]
        Tt=[]
        CN=[]
        TmVt=[]
        X=[]
        Y=[]
        Z=[]
        VX=[]
        VY=[]
        VZ=[]
        FX=[]
        FY=[]
        FZ=[]
        FVX=[]
        FVY=[]
        FVZ=[]
        Adsorbed=[]
        for pdata in InData:
            (id,tt,cn,tmvt,x,y,z,vx,vy,vz,fx,fy,fz,fvx,fvy,fvz,ad)=pdata.split('\t',16)
            ID.append(int(id))
            Tt.append(int(tt))
            CN.append(int(cn))
            TmVt.append(int(tmvt))
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
            VX.append(float(vx))
            VY.append(float(vy))
            VZ.append(float(vz))
            FX.append(float(fx))
            FY.append(float(fy))
            FZ.append(float(fz))
            FVX.append(float(fvx))
            FVY.append(float(fvy))
            FVZ.append(float(fvz))
            if(ad=='False\n'):
                Adsorbed.append(0)
            elif(ad=='True\n'):
                Adsorbed.append(1)
    ID=np.array(ID)
    Tt=np.array(Tt)
    CN=np.array(CN)
    TmVt=np.array(TmVt)
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    VX=np.array(VX)
    VY=np.array(VY)
    VZ=np.array(VZ)
    FX=np.array(FX)
    FY=np.array(FY)
    FZ=np.array(FZ)
    FVX=np.array(FVX)
    FVY=np.array(FVY)
    FVZ=np.array(FVZ)
    Adsorbed=np.array(Adsorbed)
    #
    return(ID,Tt,CN,TmVt,X,Y,Z,VX,VY,VZ,FX,FY,FZ,FVX,FVY,FVZ,Adsorbed)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Collision_Time(N,X,Y,VZ,FVZ,mpVelocity,Adsorbed,I,XSN=10,XL=0.0,XH=0.0,F=1.0):
    #
    d=5
    #
    if(XL!=XH):
        XLow=XL
        XHigh=XH
    else:
        XLow=np.min(X)
        XHigh=np.max(X)
    dX=(XHigh-XLow)/(I*XSN)
    #
    x=np.zeros((XSN,I+1))
    MinCollision=np.zeros((XSN,I+1))
    MaxCollision=np.zeros((XSN,I+1))
    AveCollision=np.zeros((XSN,I+1))
    for j in range(XSN):
        for i in range(I+1):
            count=0
            for n in range(N):
                #if(Adsorbed[n]!=1 and XLow+(j*I+i-0.5)*dX<=X[n]<XHigh-((XSN-j)*I-i-0.5)*dX):
                #if(Adsorbed[n]!=1 and XLow+(j*I+i-0.5)*dX<=X[n]-2*d/abs(VZ[n])<XHigh-((XSN-j)*I-i-0.5)*dX):
                if(Adsorbed[n]!=1 and XLow+(j*I+i-0.5)*dX<=X[n]-d/abs(VZ[n])-d/abs(FVZ[n])<XHigh-((XSN-j)*I-i-0.5)*dX):
                #if(Adsorbed[n]!=1 and XLow+(j*I+i-0.5)*dX<=X[n]-d/abs(VZ[n])-d/mpVelocity<XHigh-((XSN-j)*I-i-0.5)*dX):
                    count+=1
                    AveCollision[j,i]+=Y[n]
                    if(Y[n]<MinCollision[j,i] or MinCollision[j,i]==0):
                        MinCollision[j,i]=Y[n]
                    if(Y[n]>MaxCollision[j,i]):
                        MaxCollision[j,i]=Y[n]
            x[j,i]=(XLow+XHigh+(2*i-(XSN-2*j)*I)*dX)/2
            x[j,i]/=np.sqrt(1/F)
            if(count!=0):
                AveCollision[j,i]/=count
    #
    return(x,MinCollision,MaxCollision,AveCollision)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Criterion(N,UnA_N,X,Adsorbed,I,XSN=10,XL=0.0,XH=0.0):
    #
    if(XL!=XH):
        XLow=XL
        XHigh=XH
    else:
        XLow=np.min(X)
        XHigh=np.max(X)
    dX=(XHigh-XLow)/(I*XSN)
    #
    x=np.zeros((XSN,I+1))
    CollisionCount_UnA=np.zeros((XSN,I+1))
    CriterionCollision_UnA=np.zeros((XSN,I+1))
    CollisionCount_Ad=np.zeros((XSN,I+1))
    CriterionCollision_Ad=np.zeros((XSN,I+1))
    for j in range(XSN):
        #
        for i in range(I+1):
            #
            count1=0
            count2=0
            for n in range(N):
                if(Adsorbed[n]!=1 and XLow+(j*I+i-0.5)*dX<=X[n]<XHigh-((XSN-j)*I-i-0.5)*dX):
                    count1+=1
                elif(Adsorbed[n]==1 and XLow+(j*I+i-0.5)*dX<=X[n]<XHigh-((XSN-j)*I-i-0.5)*dX):
                    count2+=1
            x[j,i]=(XLow+XHigh+(2*i-(XSN-2*j)*I)*dX)/2
            CollisionCount_UnA[j,i]=count1
            CriterionCollision_UnA[j,i]=count1/(UnA_N*dX)
            CollisionCount_Ad[j,i]=count2
            CriterionCollision_Ad[j,i]=count2/((N-UnA_N)*dX)
    #
    return(x,CollisionCount_UnA,CriterionCollision_UnA,CollisionCount_Ad,CriterionCollision_Ad)
################################################################################
def Criterion_Plot(XName,YName,LineName1,LineName2,x,f1,f2,XSN,YL,Y1H,Y2H,f3=np.zeros(1)):
    for j in range(XSN):
        xl=int(x[j,0])
        xh=int(x[j,-1])
        yl=YL
        y1h=min(Y1H,1.25*max(f1[j,:])-0.25*yl)
        y2h=min(Y2H,1.25*max(f2[j,:])-0.25*yl)
        yh=max(y1h,y2h)
        #
        fig,ax1=mpl.subplots()
        lab1=LineName1
        lab2=LineName2
        ax1.set_xlabel('$'+XName+'$')
        ax1.set_xlim(xl,xh)
        ax1.set_ylabel('$'+YName+'$')
        ax1.set_ylim(yl,yh)
        ax1.plot(x[j,:],f1[j,:],'ro',markersize=4,label=lab1)
        ax1.plot(x[j,:],f2[j,:],'bv',markersize=4,label=lab2)
        if(f3.size>1):
            ax1.plot(x[j,:],f3[j,:],'g',markersize=4,label='Average')
        ax1.legend(loc='upper right',fontsize='x-small')
        #ax1.savefig(XName+'-'+YName+'_Segment_'+str(j)+'.png',dpi=600)
        #ax1.close()
        #
        #fig=mpl.figure()
        #lab='UnAdsorbed-IntTime Distribution'
        #mpl.plot(x[j,:],f[j,:],'ro',markersize=4,label=lab)
        #mpl.legend(loc='upper right',fontsize='x-small')
        #mpl.xlabel('$'+str(Name)+'$')
        #mpl.ylabel('$f$')
        #mpl.axis([xl,xh,yl,yh])
        mpl.savefig(XName+'-'+YName+'_Segment_'+str(j)+'.png',dpi=600)
        #
        #mpl.show()
        mpl.close()
################################################################################
def main():
    FileName='Incident_Reflection.data'
    (ID,Tt,CN,TmVt,X,Y,Z,VX,VY,VZ,FX,FY,FZ,FVX,FVY,FVZ,Adsorbed)=ReadFile(FileName)
    #
    GasT=300.0
    WallT=300.0
    GasTGasmpV=math.sqrt(2*1.38E-23*GasT/(39.95/6.02*1E-26))
    WallTGasmpV=math.sqrt(2*1.38E-23*WallT/(39.95/6.02*1E-26))#!=WallTWallmpV
    GasTGasmpV/=math.sqrt(5.207E-20/(195.08/6.02*1E-26))
    WallTGasmpV/=math.sqrt(5.207E-20/(195.08/6.02*1E-26))
    print(GasTGasmpV,WallTGasmpV)
    VX/=GasTGasmpV
    VY/=GasTGasmpV
    VZ/=GasTGasmpV
    FVX/=GasTGasmpV
    FVY/=GasTGasmpV
    FVZ/=GasTGasmpV
    #
    N=len(ID)
    for i in range(N):
        Tt[i]=Tt[i]*0.001
    UnA_N=0
    Ad_N=0
    for i in range(len(Adsorbed)):
        if(Adsorbed[i]==0):
            UnA_N+=1
        else:
            Ad_N+=1
    print(UnA_N,Ad_N)
    #
    I=200
    XSN=6
    #
    cri_Coll,Coll_Count_UnA,cri_f_UnA,Coll_Count_Ad,cri_f_Ad=Criterion(N,UnA_N,CN,Adsorbed,I,XSN,0.0,1200.0)
    Criterion_Plot('Number of Collision','f','UnAdsorbed','Adsorbed',cri_Coll,cri_f_UnA,cri_f_Ad,XSN,0.0,100.0,100.0)
    Criterion_Plot('Number of Collision','CountN','UnAdsorbed','Adsorbed',cri_Coll,Coll_Count_UnA,Coll_Count_Ad,XSN,0.0,1000000.0,1000000.0)
    I=400
    XSN=1
    F=0.1
    cri_Time,MinColl,MaxColl,AveColl=Collision_Time(N,Tt,CN,VZ*GasTGasmpV,FVZ*GasTGasmpV,WallTGasmpV,Adsorbed,I,XSN,-500.0,500.0,F)
    with open('Min-Max-Ave.data','w') as out:
        print('%s\t%s\t%s\t%s'%('Time','MinColl','MaxColl','AveColl'),file=out)
        for i in range(np.size(cri_Time,0)):
            for j in range(np.size(cri_Time,1)):
                if(MaxColl[i,j]!=0):
                    print('%f\t%f\t%f\t%f'%(cri_Time[i,j],MinColl[i,j],MaxColl[i,j],AveColl[i,j]),file=out)
    Criterion_Plot('Interaction Time-No Interaction Time','Number of Collision','Min','Max',cri_Time,MinColl,MaxColl,XSN,0.0,1000.0,1000.0,AveColl)
################################################################################
if __name__ == '__main__':
    #
    main()
