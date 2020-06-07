import numpy as np
import numba as nb
import math
import matplotlib.pyplot as mpl
################################################################################
def ReadFile(FileName):
    #
    with open(FileName,'r') as In:
        Data=In.readlines()
        InData=Data[1:]
        ID=[]
        Tt=[]
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
            (id,tt,x,y,z,vx,vy,vz,fx,fy,fz,fvx,fvy,fvz,ad)=pdata.split('\t',14)
            ID.append(int(id))
            Tt.append(float(tt))
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
    return(ID,Tt,X,Y,Z,VX,VY,VZ,FX,FY,FZ,FVX,FVY,FVZ,Adsorbed)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Distribution_Cloud(N,UnA_N,X,Y,Adsorbed,I,J,XL=0.0,XH=0.0,YL=0.0,YH=0.0):
    #
    if(XL!=XH):
        XLow=XL
        XHigh=XH
    else:
        XLow=np.min(X)
        XHigh=np.max(X)
    if(YL!=YH):
        YLow=YL
        YHigh=YH
    else:
        YLow=np.min(Y)
        YHigh=np.max(Y)
    dX=(XHigh-XLow)/I
    dY=(YHigh-YLow)/J
    #
    x=np.zeros((I+1,3))
    y=np.zeros((J+1,3))
    Rf=np.zeros((I+1,J+1))
    for i in range(I+1):
        x[i,0]=XLow+(i-0.5)*dX
        x[i,1]=(XLow+XHigh+(2*i-I)*dX)/2
        x[i,2]=XHigh-(I-i-0.5)*dX
    for j in range(J+1):
        y[j,0]=YLow+(j-0.5)*dY
        y[j,1]=(YLow+YHigh+(2*j-J)*dY)/2
        y[j,2]=YHigh-(J-j-0.5)*dY
    #
    for j in range(J+1):
        for i in range(I+1):
            count=0
            for n in range(N):
                if(Adsorbed[n]!=1 and x[i,0]<=X[n]<x[i,2] and y[j,0]<=Y[n]<y[j,2]):
                    count+=1
            Rf[i,j]=count/(UnA_N*dX*dY)
    #
    return(x,y,Rf)
################################################################################
def Cloud_dat(Name1,Name2,I,J,x,y,Rf,Type='MD',Loop=0):
    #
    if(Type=='MD'):
        with open(Name1+'-'+Name2+'.dat','w') as Out:
            print('TITLE="Distribution_Cloud"',file=Out)
            print('VARIABLES="'+Name1+'","'+Name2+'","f"',file=Out)
            print('ZONE I='+str(I+1)+', J='+str(J+1)+', F=POINT',file=Out)
            for j in range(J+1):
                for i in range(I+1):
                    print(x[i,1],y[j,1],Rf[i,j],file=Out)
    elif(Type=='CLL'):
        with open(Name1+'-'+Name2+'_Loop'+str(Loop)+'.dat','w') as Out:
            print('TITLE="Distribution_Cloud"',file=Out)
            print('VARIABLES="'+Name1+'","'+Name2+'","f"',file=Out)
            print('ZONE I='+str(I+1)+', J='+str(J+1)+', F=POINT',file=Out)
            for j in range(J+1):
                for i in range(I+1):
                    print(x[i,1],y[j,1],Rf[i,j],file=Out)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Distribution(N,UnA_N,X1,X2,Adsorbed,I,XL=0.0,XH=0.0):
    #
    if(XL!=XH):
        XLow=XL
        XHigh=XH
    else:
        XLow1=np.min(X1)
        XLow2=np.min(X2)
        XLow=min(XLow1,XLow2)
        XHigh1=np.max(X1)
        XHigh2=np.max(X2)
        XHigh=max(XHigh1,XHigh2)
    dX=(XHigh-XLow)/I
    #
    x=np.zeros(I+1)
    f1=np.zeros(I+1)
    f2=np.zeros(I+1)
    f3=np.zeros(I+1)
    for i in range(I+1):
        count1=0
        count2=0
        count3=0
        for n in range(N):
            if(XLow+(i-0.5)*dX<=X1[n]<XHigh-(I-i-0.5)*dX):
                count1+=1
            if(Adsorbed[n]!=1 and XLow+(i-0.5)*dX<=X2[n]<XHigh-(I-i-0.5)*dX):
                count2+=1
            if(Adsorbed[n]==1 and XLow+(i-0.5)*dX<=X1[n]<XHigh-(I-i-0.5)*dX):
                count3+=1
        x[i]=(XLow+XHigh+(2*i-I)*dX)/2
        f1[i]=count1/(N*dX)
        f2[i]=count2/(UnA_N*dX)
        f3[i]=count3/((N-UnA_N)*dX)
    #
    return(x,f1,f2,f3)
################################################################################
def Distribution_Plot(Name,x,f1,f2,f3,f4,f5,f6,XL,XH,YL,YH):
    fig=mpl.figure()
    mpl.plot(x,f1,'ro',markersize=4,label='Incident Velocity Distribution')
    mpl.plot(x,f2,'r--',markersize=4,label='CLL Incident Velocity Distribution')
    mpl.plot(x,f3,'go',markersize=4,label='Reflection Velocity Distribution')
    mpl.plot(x,f4,'g--',markersize=4,label='CLL Reflection Velocity Distribution')
    mpl.plot(x,f5,'bo',markersize=4,label='Incident Velocity-Adsorbed Distribution')
    mpl.plot(x,f6,'b--',markersize=4,label='Nor')
    mpl.legend(loc='upper right',fontsize='x-small')
    mpl.xlabel('$'+str(Name)+'$')
    mpl.ylabel('$f$')
    mpl.axis([XL,XH,YL,YH])
    mpl.savefig('IRA_f_'+Name+'.png',dpi=600)
    #
    #mpl.show()
    mpl.close()
################################################################################
@nb.jit(nopython=True,nogil=True)
def CLL_R(ui,ur,evi,ewi,l1,l2,l3):
    tt=l1*math.exp(-l2*evi-l3*ewi)
    R=1/math.sqrt(np.pi*tt*(2-tt))*math.exp(-(ur-(1-tt)*ui)**2/(tt*(2-tt)))
    return(R)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Gradient_Least_Squares(N,ui,ur,Rfuir,fui,VX,FVX,VY,VZ,mpV,lt1,lt2,lt3):
    #
    CLL_Rfu=np.zeros((len(ui),len(ur)))
    CLL_Rfu1=np.zeros((len(ui),len(ur)))
    CLL_Rfu2=np.zeros((len(ui),len(ur)))
    CLL_Rfu3=np.zeros((len(ui),len(ur)))
    CLL_fui=np.zeros(len(ui))
    CLL_fur=np.zeros(len(ur))
    CLL_fu=0.0
    CLL_Nor=np.zeros(len(ui))
    Err=0.0
    Err1=0.0
    Err2=0.0
    Err3=0.0
    alpha1=0.01
    alpha2=0.01
    alpha3=0.01
    dlt1=0.1*alpha1
    dlt2=0.1*alpha2
    dlt3=0.1*alpha3
    for i in range(len(ui)):
        CountNi=0
        AverOthV1=0.0
        AverOthV2=0.0
        for n in range(N):
            if(ui[i,0]<=VX[n]<ui[i,2]):
                CountNi+=1
                AverOthV1+=VY[n]**2
                AverOthV2+=VZ[n]**2
        if(CountNi!=0):
            AverOthV1/=CountNi
            AverOthV2/=CountNi
            dfui=fui[i]
            for j in range(len(ur)):
                CLL_Rfu[i,j]=0.0
                CLL_Rfu1[i,j]=0.0
                CLL_Rfu2[i,j]=0.0
                CLL_Rfu3[i,j]=0.0
                CountNij=0
                for n in range(N):
                    if(ui[i,0]<=VX[n]<ui[i,2] and ur[j,0]<=FVX[n]<ur[j,2]):
                        CountNij+=1
                        CLL_Rfu[i,j]+=CLL_R(VX[n]/mpV,FVX[n]/mpV,AverOthV1/(mpV**2),AverOthV2/(mpV**2),lt1,lt2,lt3)/mpV*dfui
                        CLL_Rfu1[i,j]+=CLL_R(VX[n]/mpV,FVX[n]/mpV,AverOthV1/(mpV**2),AverOthV2/(mpV**2),lt1+dlt1,lt2,lt3)/mpV*dfui
                        CLL_Rfu2[i,j]+=CLL_R(VX[n]/mpV,FVX[n]/mpV,AverOthV1/(mpV**2),AverOthV2/(mpV**2),lt1,lt2+dlt2,lt3)/mpV*dfui
                        CLL_Rfu3[i,j]+=CLL_R(VX[n]/mpV,FVX[n]/mpV,AverOthV1/(mpV**2),AverOthV2/(mpV**2),lt1,lt2,lt3+dlt3)/mpV*dfui
                if(CountNij!=0):
                    CLL_Rfu[i,j]/=CountNij
                    CLL_Rfu1[i,j]/=CountNij
                    CLL_Rfu2[i,j]/=CountNij
                    CLL_Rfu3[i,j]/=CountNij
                CLL_fui[i]+=CLL_Rfu[i,j]*(ur[j,2]-ur[j,0])
                CLL_Nor[i]+=CLL_Rfu[i,j]/dfui*(ur[j,2]-ur[j,0])
                CLL_fur[j]+=CLL_Rfu[i,j]*(ui[i,2]-ui[i,0])
                CLL_fu+=CLL_Rfu[i,j]*(ui[i,2]-ui[i,0])*(ur[j,2]-ur[j,0])
                Err+=(Rfuir[i,j]-CLL_Rfu[i,j])**2
                Err1+=(Rfuir[i,j]-CLL_Rfu1[i,j])**2
                Err2+=(Rfuir[i,j]-CLL_Rfu2[i,j])**2
                Err3+=(Rfuir[i,j]-CLL_Rfu3[i,j])**2
    dErr_dlt1=(Err1-Err)/dlt1
    dErr_dlt2=(Err2-Err)/dlt2
    dErr_dlt3=(Err3-Err)/dlt3
    lt1=lt1-alpha1*dErr_dlt1
    if(lt1<0.0):
        lt1=2*dlt1
    elif(lt1>2.0):
        lt1=2.0-2*dlt1
    lt2=lt2-alpha2*dErr_dlt2
    lt3=lt3-alpha3*dErr_dlt3
    return(CLL_Rfu,CLL_Nor,CLL_fu,CLL_fui,CLL_fur,Err,lt1,lt2,lt3)
################################################################################
def Err_Plot(Name,X_l,Y_Err,Y_lt1,Y_lt2,Y_lt3):
    #
    fig1=mpl.figure()
    mpl.plot(X_l,Y_Err,'ro',markersize=4,label='Last Error='+str(Y_Err[-1]))
    mpl.legend(loc='upper right',fontsize='x-small')
    mpl.xlabel('$Loop$')
    mpl.ylabel('$Error$')
    #mpl.axis([XLow,XHigh,YL,YH])
    mpl.savefig(Name+'_Error_Loop.png',dpi=600)
    #
    fig2=mpl.figure()
    mpl.plot(X_l,Y_lt1,'ro',markersize=4,label='Last lt1='+str(Y_lt1[-1]))
    mpl.plot(X_l,Y_lt2,'bo',markersize=4,label='Last lt2='+str(Y_lt2[-1]))
    mpl.plot(X_l,Y_lt3,'go',markersize=4,label='Last lt3='+str(Y_lt3[-1]))
    mpl.legend(loc='upper right',fontsize='x-small')
    mpl.xlabel('$Loop$')
    mpl.ylabel('$lt$')
    #mpl.axis([XLow,XHigh,YL,YH])
    mpl.savefig(Name+'_lt_Loop.png',dpi=600)
    #
    #mpl.show()
    mpl.close()
################################################################################
def main():
    FileName='Incident_Reflection.data'
    (ID,Tt,X,Y,Z,VX,VY,VZ,FX,FY,FZ,FVX,FVY,FVZ,Adsorbed)=ReadFile(FileName)
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
    I=50
    ui,tu,Rfuit=Distribution_Cloud(N,UnA_N,VX,Tt,Adsorbed,I,I,-3.0,3.0,0.0,50.0)
    Cloud_dat('VX','Tt',I,I,ui,tu,Rfuit)
    vi,tv,Rfvit=Distribution_Cloud(N,UnA_N,VY,Tt,Adsorbed,I,I,-3.0,3.0,0.0,50.0)
    Cloud_dat('VY','Tt',I,I,vi,tv,Rfvit)
    wi,tw,Rfwit=Distribution_Cloud(N,UnA_N,VZ,Tt,Adsorbed,I,I,-3.0,0.0,0.0,50.0)
    Cloud_dat('VZ','Tt',I,I,wi,tw,Rfwit)
    #
    ui,vr,Rfuivr=Distribution_Cloud(N,UnA_N,VX,FVY,Adsorbed,I,I,-3.0,3.0,-3.0,3.0)
    Cloud_dat('VX','FVY',I,I,ui,vr,Rfuivr)
    ui,wr,Rfuiwr=Distribution_Cloud(N,UnA_N,VX,FVZ,Adsorbed,I,I,-3.0,3.0,0.0,3.0)
    Cloud_dat('VX','FVZ',I,I,ui,wr,Rfuiwr)
    #
    ui,ur,Rfuir=Distribution_Cloud(N,UnA_N,VX,FVX,Adsorbed,I,I,-3.0,3.0,-3.0,3.0)
    Cloud_dat('VX','FVX',I,I,ui,ur,Rfuir)
    vi,vr,Rfvir=Distribution_Cloud(N,UnA_N,VY,FVY,Adsorbed,I,I,-3.0,3.0,-3.0,3.0)
    Cloud_dat('VY','FVY',I,I,vi,vr,Rfvir)
    wi,wr,Rfwir=Distribution_Cloud(N,UnA_N,VZ,FVZ,Adsorbed,I,I,-3.0,0.0,0.0,3.0)
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir)
    #
    u,fui,fur,fua=Distribution(N,UnA_N,VX,FVX,Adsorbed,I,-3.0,3.0)
    v,fvi,fvr,fva=Distribution(N,UnA_N,VY,FVY,Adsorbed,I,-3.0,3.0)
    #Distribution_Plot('VY',v,fvi,fvr,fva,-3.0,3.0,0.0,1.0)
    w,fwi,fwr,fwa=Distribution(N,UnA_N,VZ,FVZ,Adsorbed,I,-3.0,3.0)
    #Distribution_Plot('VZ',w,fwi,fwr,fwa,-3.0,3.0,0.0,1.4)
    #
    mpV=math.sqrt(2*1.38E-23*300.0/(39.95/6.02*1E-26))
    mpV/=math.sqrt(5.207E-20/(195.08/6.02*1E-26))
    print(mpV)
    #
    lt1=1.45
    lt2=0.88
    lt3=0.88
    Loops=0
    l_X=np.zeros(Loops+1)
    Err_X=np.zeros(Loops+1)
    lt1_X=np.zeros(Loops+1)
    lt2_X=np.zeros(Loops+1)
    lt3_X=np.zeros(Loops+1)
    #
    for l in range(Loops+1):
        CLL_Rfu,CLL_Nor,CLL_fu,CLL_fui,CLL_fur,Err,lt1,lt2,lt3=Gradient_Least_Squares(N,ui,ur,Rfuir,fui,VX,FVX,VY,VZ,mpV,lt1,lt2,lt3)
        print(l,Err,lt1,lt2,lt3,CLL_fu)
        l_X[l]=l
        Err_X[l]=Err
        lt1_X[l]=lt1
        lt2_X[l]=lt2
        lt3_X[l]=lt3
        if(l%10==0):
            Cloud_dat('VX','CLL_FVX',I,I,ui,ur,CLL_Rfu,'CLL',l)
    #
    Err_Plot('X',l_X,Err_X,lt1_X,lt2_X,lt3_X)
    Distribution_Plot('VX',u,fui,CLL_fui,fur,CLL_fur,fua,CLL_Nor,-3.0,3.0,0.0,1.0)
################################################################################
if __name__ == '__main__':
    #
    main()
