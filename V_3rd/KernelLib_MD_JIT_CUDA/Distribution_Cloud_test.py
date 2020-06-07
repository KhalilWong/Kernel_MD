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
def Random_Sampling(N,SamplesN,ID,VX,VY,VZ,FVX,FVY,FVZ,Adsorbed):
    #
    #SamplesID=random.sample(range(N),SamplesN)
    SamplesID=np.random.choice(N,SamplesN,replace=False)
    SampleID=[]
    SampleVX=[]
    SampleVY=[]
    SampleVZ=[]
    SampleFVX=[]
    SampleFVY=[]
    SampleFVZ=[]
    SampleAds=[]
    SamplesUAN=0
    for i in range(SamplesN):
        IndexID=np.where(ID==SamplesID[i])
        SampleID.append(int(ID[IndexID]))
        SampleVX.append(float(VX[IndexID]))
        SampleVY.append(float(VY[IndexID]))
        SampleVZ.append(float(VZ[IndexID]))
        SampleFVX.append(float(FVX[IndexID]))
        SampleFVY.append(float(FVY[IndexID]))
        SampleFVZ.append(float(FVZ[IndexID]))
        SampleAds.append(int(Adsorbed[IndexID]))
        if(int(Adsorbed[IndexID])==0):
            SamplesUAN+=1
    SampleID=np.array(SampleID)
    SampleVX=np.array(SampleVX)
    SampleVY=np.array(SampleVY)
    SampleVZ=np.array(SampleVZ)
    SampleFVX=np.array(SampleFVX)
    SampleFVY=np.array(SampleFVY)
    SampleFVZ=np.array(SampleFVZ)
    SampleAds=np.array(SampleAds)
    #
    return(SamplesUAN,SampleID,SampleVX,SampleVY,SampleVZ,SampleFVX,SampleFVY,SampleFVZ,SampleAds)
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
@nb.jit(nopython=True,nogil=True)
def Distribution_Cloud_Energy(N,UnA_N,X,Y,Adsorbed,I,J,XL=0.0,XH=0.0,YL=0.0,YH=0.0,Ev1in=np.zeros((1,1)),Ev2in=np.zeros((1,1)),Ev1out=np.zeros((1,1)),Ev2out=np.zeros((1,1))):
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
    E0ExcMat=np.zeros((I+1,J+1))
    E1ExcMat=np.zeros((I+1,J+1))
    ETExcMat=np.zeros((I+1,J+1))
    ERateInMat=np.zeros((I+1,J+1))
    ERateOutMat=np.zeros((I+1,J+1))
    ERateExcMat=np.zeros((I+1,J+1))
    #E2ExcMat=np.zeros((I+1,J+1))
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
            sumE0in=0.0
            sumE1in=0.0
            sumE2in=0.0
            sumE0out=0.0
            sumE1out=0.0
            sumE2out=0.0
            for n in range(N):
                if(Adsorbed[n]!=1 and x[i,0]<=X[n]<x[i,2] and y[j,0]<=Y[n]<y[j,2]):
                    count+=1
                    sumE0in+=X[n]**2
                    sumE1in+=Ev1in[n]**2
                    sumE2in+=Ev2in[n]**2
                    sumE0out+=Y[n]**2
                    sumE1out+=Ev1out[n]**2
                    sumE2out+=Ev2out[n]**2
            Rf[i,j]=count/(UnA_N*dX*dY)
            if(count!=0):
                E0ExcMat[i,j]=(sumE0out-sumE0in)/count
                E1ExcMat[i,j]=(sumE1out+sumE2out-sumE1in-sumE2in)/count
                ETExcMat[i,j]=E0ExcMat[i,j]+E1ExcMat[i,j]
                ERateInMat[i,j]=math.sqrt(sumE0in/(sumE0in+sumE1in+sumE2in))
                ERateOutMat[i,j]=math.sqrt(sumE0out/(sumE0out+sumE1out+sumE2out))
                ERateExcMat[i,j]=ERateInMat[i,j]*ERateOutMat[i,j]-math.sqrt(1-ERateInMat[i,j]**2)*math.sqrt(1-ERateOutMat[i,j]**2)
                #ERateMat[i,j]=math.sqrt(math.exp(math.log(max(x[i,1]**2,1e-8))-math.log((sumE1in+sumE2in)/count+x[i,1]**2)))
                #E2ExcMat[i,j]=(sumE2out-sumE2in)/count
            else:
                E0ExcMat[i,j]=0.0
                E1ExcMat[i,j]=0.0
                ETExcMat[i,j]=0.0
                ERateInMat[i,j]=-1.0
                ERateOutMat[i,j]=-1.0
                ERateExcMat[i,j]=-2.0
                #ERateMat[i,j]=math.sqrt(math.exp(math.log(max(x[i,1]**2,1e-8))-math.log(max(sumE1in+sumE2in+x[i,1]**2,1e-8))))
                #E2ExcMat[i,j]=sumE2out-sumE2in
    #
    return(x,y,Rf,E0ExcMat,E1ExcMat,ETExcMat,ERateInMat,ERateOutMat,ERateExcMat)
################################################################################
def Cloud_dat(Name1,Name2,I,J,x,y,Rf,Type='MD',Loop=0,Rf_Basis=np.zeros((1,1)),TailName='Tail'):
    #
    if(Type=='MD'):
        with open(Name1+'-'+Name2+'.dat','w') as Out:
            print('TITLE="Distribution_Cloud"',file=Out)
            print('VARIABLES="'+Name1+'","'+Name2+'","f_MD"',file=Out)
            print('ZONE I='+str(I+1)+', J='+str(J+1)+', F=POINT',file=Out)
            for j in range(J+1):
                for i in range(I+1):
                    print(x[i,1],y[j,1],Rf[i,j],file=Out)
    elif(Type=='CLL'):
        with open(Name1+'-'+Name2+'_Loop'+str(Loop)+'.dat','w') as Out:
            print('TITLE="Distribution_Cloud"',file=Out)
            print('VARIABLES="'+Name1+'","'+Name2+'","f","f_MD"',file=Out)
            print('ZONE I='+str(I+1)+', J='+str(J+1)+', F=POINT',file=Out)
            for j in range(J+1):
                for i in range(I+1):
                    print(x[i,1],y[j,1],Rf[i,j],Rf_Basis[i,j],file=Out)
    elif(Type=='AC'):
        with open(Name1+'-'+Name2+'-'+Type+'_Loop'+str(Loop)+'.dat','w') as Out:
            print('TITLE="Distribution_Cloud"',file=Out)
            print('VARIABLES="'+Name1+'","'+Name2+'","AC","f_MD"',file=Out)
            print('ZONE I='+str(I+1)+', J='+str(J+1)+', F=POINT',file=Out)
            for j in range(J+1):
                for i in range(I+1):
                    print(x[i,1],y[j,1],Rf[i,j],Rf_Basis[i,j],file=Out)
    elif(Type=='Sample'):
        with open(Name1+'-'+Name2+'.dat','w') as Out:
            print('TITLE="Distribution_Cloud"',file=Out)
            print('VARIABLES="'+Name1+'","'+Name2+'","f_MD","f_Sample"',file=Out)
            print('ZONE I='+str(I+1)+', J='+str(J+1)+', F=POINT',file=Out)
            for j in range(J+1):
                for i in range(I+1):
                    print(x[i,1],y[j,1],Rf[i,j],Rf_Basis[i,j],file=Out)
    elif(Type=='Energy'):
        with open(Name1+'-'+Name2+'-'+TailName+'.dat','w') as Out:
            print('TITLE="Distribution_Cloud"',file=Out)
            print('VARIABLES="'+Name1+'","'+Name2+'","f","'+TailName+'"',file=Out)
            print('ZONE I='+str(I+1)+', J='+str(J+1)+', F=POINT',file=Out)
            for j in range(J+1):
                for i in range(I+1):
                    print(x[i,1],y[j,1],Rf[i,j],Rf_Basis[i,j],file=Out)
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
    f4=np.zeros(I+1)
    for i in range(I+1):
        count1=0
        count2=0
        count3=0
        count4=0
        for n in range(N):
            if(XLow+(i-0.5)*dX<=X1[n]<XHigh-(I-i-0.5)*dX):
                count1+=1
            if(Adsorbed[n]!=1 and XLow+(i-0.5)*dX<=X2[n]<XHigh-(I-i-0.5)*dX):
                count2+=1
            if(Adsorbed[n]==1 and XLow+(i-0.5)*dX<=X1[n]<XHigh-(I-i-0.5)*dX):
                count3+=1
            if(Adsorbed[n]!=1 and XLow+(i-0.5)*dX<=X1[n]<XHigh-(I-i-0.5)*dX):
                count4+=1
        x[i]=(XLow+XHigh+(2*i-I)*dX)/2
        f1[i]=count1/(N*dX)
        f2[i]=count2/(UnA_N*dX)
        if(N-UnA_N!=0):
            f3[i]=count3/((N-UnA_N)*dX)
        else:
            f3[i]=0.0
        f4[i]=count4/(UnA_N*dX)
    #
    return(x,f1,f2,f3,f4)
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
    TimeCount=np.zeros((XSN,I+1))
    CriterionTime=np.zeros((XSN,I+1))
    #CriterionCollision=np.zeros(I+1)
    for j in range(XSN):
        #
        for i in range(I+1):
            count=0
            #count2=0
            for n in range(N):
                if(Adsorbed[n]!=1 and XLow+(j*I+i-0.5)*dX<=X[n]<XHigh-((XSN-j)*I-i-0.5)*dX):
                    count+=1
            x[j,i]=(XLow+XHigh+(2*i-(XSN-2*j)*I)*dX)/2
            TimeCount[j,i]=count
            CriterionTime[j,i]=count/(UnA_N*dX)
    #
    return(x,TimeCount,CriterionTime)
################################################################################
def Distribution_Plot(Name,x,f1,f2,f3,f4,f5,f6,f7,XL,XH,YL,YH,GasT):
    fig=mpl.figure()
    if(Name=='VZ'):
        lab1='Incident Velocity Distribution with T='+str(int((abs(x[np.where(f1==np.max(f1))])*np.sqrt(2))**2*GasT))+' K'
        lab2='Reflection Velocity Distribution with T='+str(int((abs(x[np.where(f2==np.max(f2))])*np.sqrt(2))**2*GasT))+' K'
        if(np.max(f3)!=0.0):
            lab3='Incident Velocity-Adsorbed Distribution with T='+str(int((abs(x[np.where(f3==np.max(f3))[0][0]])*np.sqrt(2))**2*GasT))+' K'
        else:
            lab3='Incident Velocity-Adsorbed Distribution'
        lab4='Incident Velocity-UnAdsorbed Distribution with T='+str(int((abs(x[np.where(f4==np.max(f4))])*np.sqrt(2))**2*GasT))+' K'
        lab5='CLL Incident Velocity Distribution with T='+str(int((abs(x[np.where(f5==np.max(f5))])*np.sqrt(2))**2*GasT))+' K'
        lab6='CLL Reflection Velocity Distribution with T='+str(int((abs(x[np.where(f6==np.max(f6))[0][0]+len(f6)-1])*np.sqrt(2))**2*GasT))+' K'
        x5=x[:len(f5)]
        x6=x[len(f6)-1:]
        x7=x[:len(f7)]
    else:
        lab1='Incident Velocity Distribution with T='+str(int((1/(np.max(f1)*np.sqrt(np.pi)))**2*300))+' K'
        lab2='Reflection Velocity Distribution with T='+str(int((1/(np.max(f2)*np.sqrt(np.pi)))**2*300))+' K'
        lab3='Incident Velocity-Adsorbed Distribution with T='+str(int((1/(np.max(f3)*np.sqrt(np.pi)))**2*300))+' K'
        lab4='Incident Velocity-UnAdsorbed Distribution with T='+str(int((1/(np.max(f4)*np.sqrt(np.pi)))**2*300))+' K'
        lab5='CLL Incident Velocity Distribution with T='+str(int((1/(np.max(f5)*np.sqrt(np.pi)))**2*300))+' K'
        lab6='CLL Reflection Velocity Distribution with T='+str(int((1/(np.max(f6)*np.sqrt(np.pi)))**2*300))+' K'
        x5=x
        x6=x
        x7=x
    mpl.plot(x,f1,'ro',markersize=4,label=lab1)
    mpl.plot(x,f2,'go',markersize=4,label=lab2)
    mpl.plot(x,f3,'bo',markersize=4,label=lab3)
    mpl.plot(x,f4,'mo',markersize=4,label=lab4)
    mpl.plot(x5,f5,'r--',markersize=4,label=lab5)
    mpl.plot(x6,f6,'g--',markersize=4,label=lab6)
    mpl.plot(x7,f7,'m--',markersize=4,label='Nor')
    mpl.legend(loc='upper right',fontsize='xx-small')
    mpl.xlabel('$'+str(Name)+'$')
    mpl.ylabel('$f$')
    mpl.axis([XL,XH,YL,YH])
    mpl.savefig('IRA_f_'+Name+'.png',dpi=600)
    #
    #mpl.show()
    mpl.close()
################################################################################
def Criterion_Plot(Name,x,f1,f2,XSN,Y1L,Y1H,Y2L,Y2H):
    for j in range(XSN):
        xl=int(x[j,0])
        xh=int(x[j,-1])
        y1l=Y1L
        y1h=min(Y1H,1.25*max(f1[j,:])-0.25*y1l)
        y2l=Y2L
        y2h=min(Y2H,1.25*max(f2[j,:])-0.25*y2l)
        #
        fig,ax1=mpl.subplots()
        lab1='UnAdsorbed-IntTime f Distribution'
        ax1.set_xlabel('$'+str(Name)+'$')
        ax1.set_xlim(xl,xh)
        ax1.set_ylabel('$f$', color='r')
        ax1.set_ylim(y1l,y1h)
        l1=ax1.plot(x[j,:],f1[j,:],'ro',markersize=4,label=lab1)
        ax1.tick_params(axis='y', labelcolor='r')
        #
        ax2=ax1.twinx()
        lab2='UnAdsorbed-IntTime CountN Distribution'
        ax2.set_ylabel('$CountN$', color='b')
        ax1.set_ylim(y2l,y2h)
        l2=ax2.plot(x[j,:],f2[j,:],'b--',markersize=4,label=lab2)
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
        mpl.savefig('Criterion_'+Name+'_Segment_'+str(j)+'.png',dpi=600)
        #
        #mpl.show()
        mpl.close()
################################################################################
@nb.jit(nopython=True,nogil=True)
def CLL_R(Type,ui,ur,vi,wi,l1,l2,l3,l4,l5,l6):
    if(Type==0):
        #tt=l1*math.exp(-l2*vi**2-l3*wi**2)
        tt=1-math.log(l1+l2*vi**2-l3*wi**2)
        R=1/math.sqrt(np.pi*tt*(2-tt))*math.exp(-(ur-(1-tt)*ui)**2/(tt*(2-tt)))
    elif(Type==1):
        #tt=l1-l2*math.exp(-l3*abs(ur**2-ui**2))
        #tt=1-math.log(l1+l1*vi**2+l3*wi**2)
        #tt=l1+l2*np.cos(l3*(vi**2+wi**2)+l4)
        #alpha=l1*math.exp(-l2*vi**2-l2*wi**2)
        #alpha=tt*(2-tt)
        #alpha=tt
        alpha1=l1
        alpha2=l2
        alpha3=l3
        N_theta=1000
        #d_theta=np.pi/N_theta
        R=0.0
        R1=0.0
        R2=0.0
        R3=0.0
        for i in range(N_theta):
            #R+=2*ur/alpha*math.exp(-(ur**2+(1-alpha)*ui**2-2*math.sqrt(1-alpha)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/alpha)/N_theta
            R1+=2*ur/alpha1*math.exp(-(ur**2+(1-alpha1)*ui**2-2*math.sqrt(1-alpha1)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/alpha1)/N_theta
            #R2+=2*ur/alpha2*math.exp(-(ur**2+(1-alpha2)*ui**2-2*math.sqrt(1-alpha2)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/alpha2)/N_theta
            R3+=2*ur/alpha3*math.exp(-(ur**2+(1-alpha3)*ui**2-2*math.sqrt(1-alpha3)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/alpha3)/N_theta
        R=l4*R1+l6*R3
    return(R)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Reciprocal(Type,ui,ur,l1,l2,l3,l4):
    #0<l1<1;0<l2<1;l3>0;l4>0
    alpha1=l1
    alpha2=l2
    if(Type==0):
        R1=1/math.sqrt(np.pi*alpha1)*math.exp(-(ur-math.sqrt(1-alpha1)*ui)**2/alpha1)
        R2=1/math.sqrt(np.pi*alpha2)*math.exp(-(ur-math.sqrt(1-alpha2)*ui)**2/alpha2)
        R=1/(l3/R1+l4/R2)
    elif(Type==1):
        N_theta=1000
        #d_theta=np.pi/N_theta
        R1=0.0
        R2=0.0
        for i in range(N_theta):
            R1+=2*ur/alpha1*math.exp(-(ur**2+(1-alpha1)*ui**2-2*math.sqrt(1-alpha1)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/alpha1)/N_theta
            R2+=2*ur/alpha2*math.exp(-(ur**2+(1-alpha2)*ui**2-2*math.sqrt(1-alpha2)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/alpha2)/N_theta
        R=1/(l3/R1+l4/R2)
    return(R)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Epstein(Type,ui,ur,l1,l2,l3,l4):
    #l1>0;0<l2<l1;0<l3<1
    P=math.exp(-l1*ui**2)+l3*(1-math.exp(-l2*ui**2))
    a=math.exp(-l4)
    if(Type==0):
        x=ui-ur
        LimDelta=1/(a*math.sqrt(np.pi))*math.exp(-x**2/a**2)
        R=P/math.sqrt(np.pi)*math.exp(-ur**2)+(1-P)*LimDelta
    elif(Type==1):
        x=ui+ur
        LimDelta=1/(a*math.sqrt(np.pi))*math.exp(-x**2/a**2)
        R=P*2*ur*math.exp(-ur**2)+(1-P)*LimDelta
    return(R)
################################################################################
@nb.jit(nopython=True,nogil=True)
def Gradient_Least_Squares(Model,Type,N,ui,vi,wi,ur,Rfuir,fui,fvi,fwi,VX,FVX,VY,VZ,Adsorbed,GasTmpV,WallTmpV,lt1,lt2,lt3,lt4,lt5,lt6,mu,v1,v2,v3,v4,v5,v6):
    #
    Mod_Rfu=np.zeros((len(ui),len(ur)))
    Mod_Rfu1=np.zeros((len(ui),len(ur)))
    Mod_Rfu2=np.zeros((len(ui),len(ur)))
    Mod_Rfu3=np.zeros((len(ui),len(ur)))
    Mod_Rfu4=np.zeros((len(ui),len(ur)))
    #Mod_Rfu5=np.zeros((len(ui),len(ur)))
    #Mod_Rfu6=np.zeros((len(ui),len(ur)))
    Mod_fui=np.zeros(len(ui))
    Mod_fur=np.zeros(len(ur))
    Mod_Nor=np.zeros(len(ui))
    Mod_Fu=0.0
    Err=0.0
    Err1=0.0
    Err2=0.0
    Err3=0.0
    Err4=0.0
    #Err5=0.0
    #Err6=0.0
    alpha=1e-4
    dlt=1e-8
    #
    for i in range(len(ui)):
        if(fui[i]!=0):
            dfui=fui[i]
            for j in range(len(ur)):
                CountNij=0.0
                for n in range(N):
                    if(Adsorbed[n]!=1 and ui[i,0]<=VX[n]<ui[i,2] and ur[j,0]<=FVX[n]<ur[j,2]):
                        for k in range(len(vi)):
                            if(vi[k,0]<=VY[n]<vi[k,2]):
                                dfvi=fvi[k]
                                break
                        for m in range(len(wi)):
                            if(wi[m,0]<=VZ[n]<wi[m,2]):
                                dfwi=fwi[m]
                                break
                        CountNij+=dfvi*dfwi
                        Mod_Rfu[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu1[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1+dlt,lt2,lt3,lt4)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu2[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2+dlt,lt3,lt4)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu3[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3+dlt,lt4)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu4[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4+dlt)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        #Mod_Rfu5[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4,lt5+dlt,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        #Mod_Rfu6[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4,lt5,lt6+dlt)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                if(CountNij!=0):
                    Mod_Rfu[i,j]/=CountNij
                    Mod_Rfu1[i,j]/=CountNij
                    Mod_Rfu2[i,j]/=CountNij
                    Mod_Rfu3[i,j]/=CountNij
                    Mod_Rfu4[i,j]/=CountNij
                    #Mod_Rfu5[i,j]/=CountNij
                    #Mod_Rfu6[i,j]/=CountNij
                Mod_fui[i]+=Mod_Rfu[i,j]*(ur[j,2]-ur[j,0])
                Mod_Nor[i]+=Mod_Rfu[i,j]/dfui*(ur[j,2]-ur[j,0])
                Mod_fur[j]+=Mod_Rfu[i,j]*(ui[i,2]-ui[i,0])
                Mod_Fu+=Mod_Rfu[i,j]*(ui[i,2]-ui[i,0])*(ur[j,2]-ur[j,0])
                Err+=(Rfuir[i,j]-Mod_Rfu[i,j])**2
                Err1+=(Rfuir[i,j]-Mod_Rfu1[i,j])**2
                Err2+=(Rfuir[i,j]-Mod_Rfu2[i,j])**2
                Err3+=(Rfuir[i,j]-Mod_Rfu3[i,j])**2
                Err4+=(Rfuir[i,j]-Mod_Rfu4[i,j])**2
                #Err5+=(Rfuir[i,j]-Mod_Rfu5[i,j])**2
                #Err6+=(Rfuir[i,j]-Mod_Rfu6[i,j])**2
    dErr_dlt1=(Err1-Err)/dlt
    dErr_dlt2=(Err2-Err)/dlt
    dErr_dlt3=(Err3-Err)/dlt
    dErr_dlt4=(Err4-Err)/dlt
    #dErr_dlt5=(Err5-Err)/dlt
    #dErr_dlt6=(Err6-Err)/dlt
    #
    pre_v1=v1
    v1=mu*v1
    v1+=-alpha*dErr_dlt1
    lt1+=v1+mu*(v1-pre_v1)
    if(lt1<=0.0):
        lt1=dlt
        v1=-v1
    elif(lt1>=1.0-dlt):
        lt1=1.0-2*dlt
        v1=-v1
    #
    pre_v2=v2
    v2=mu*v2
    v2+=-alpha*dErr_dlt2
    lt2+=v2+mu*(v2-pre_v2)
    if(lt2<=0.0):
        lt2=dlt
        v2=-v2
    elif(lt2>=1.0-dlt):
        lt2=1.0-2*dlt
        v2=-v2
    #
    pre_v3=v3
    v3=mu*v3
    v3+=-alpha*dErr_dlt3
    lt3+=v3+mu*(v3-pre_v3)
    if(lt3<=0.0):
        lt3=dlt
        v3=-v3
    #
    pre_v4=v4
    v4=mu*v4
    v4+=-alpha*dErr_dlt4
    lt4+=v4+mu*(v4-pre_v4)
    if(lt4<=0.0):
        lt4=dlt
        v4=-v4
    #
    #pre_v5=v5
    #v5=mu*v5
    #v5+=-alpha*dErr_dlt5
    #lt5+=v5+mu*(v5-pre_v5)
    #if(lt5<0.0):
    #    lt5=dlt
    #    v5=-v5
    #
    #pre_v6=v6
    #v6=mu*v6
    #v6+=-alpha*dErr_dlt6
    #lt6+=v6+mu*(v6-pre_v6)
    #if(lt6<0.0):
    #    lt6=dlt
    #    v6=-v6

    return(Mod_Rfu,Mod_Nor,Mod_Fu,Mod_fui,Mod_fur,Err,lt1,lt2,lt3,lt4,lt5,lt6,v1,v2,v3,v4,v5,v6)
################################################################################
@nb.jit(nopython=True,nogil=True)
def CLL_O(Type,ui,ur,AC):
    if(Type==0):
        R=1/math.sqrt(np.pi*AC)*math.exp(-(ur-np.sqrt(1-AC)*ui)**2/AC)
    elif(Type==1):
        N_theta=1000
        #d_theta=np.pi/N_theta
        R=0.0
        for i in range(N_theta):
            R+=2*ur/AC*math.exp(-(ur**2+(1-AC)*ui**2-2*np.sqrt(1-AC)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/AC)/N_theta
    return(R)
################################################################################
@nb.jit(nopython=True,nogil=True)
def AC(Type,ui,ur,Rfuir,fui,GasTmpV,WallTmpV,ACMat,VMat):
    #
    CLL_Rfu=np.zeros((len(ui),len(ur)))
    CLL_Rfu_pdac=np.zeros((len(ui),len(ur)))
    CLL_Rfu_mdac=np.zeros((len(ui),len(ur)))
    Err=0.0
    alpha=1e-4
    dlt=1e-8
    mu=0.999
    for i in range(len(ui)):
        if(fui[i]!=0):
            dfui=fui[i]
            for j in range(len(ur)):
                CLL_Rfu[i,j]=CLL_O(Type,ui[i,1]*GasTmpV/WallTmpV,ur[j,1]*GasTmpV/WallTmpV,ACMat[i,j])*GasTmpV/WallTmpV*dfui
                CLL_Rfu_pdac[i,j]=CLL_O(Type,ui[i,1]*GasTmpV/WallTmpV,ur[j,1]*GasTmpV/WallTmpV,ACMat[i,j]+dlt)*GasTmpV/WallTmpV*dfui
                CLL_Rfu_mdac[i,j]=CLL_O(Type,ui[i,1]*GasTmpV/WallTmpV,ur[j,1]*GasTmpV/WallTmpV,ACMat[i,j]-dlt)*GasTmpV/WallTmpV*dfui
                Err+=(Rfuir[i,j]-CLL_Rfu[i,j])**2
                Errp=(Rfuir[i,j]-CLL_Rfu_pdac[i,j])**2
                Errm=(Rfuir[i,j]-CLL_Rfu_mdac[i,j])**2
                dErr_dlt=(Errp-Errm)/(2*dlt)
                #
                pre_v=VMat[i,j]
                VMat[i,j]=mu*VMat[i,j]
                VMat[i,j]+=-alpha*dErr_dlt
                ACMat[i,j]+=VMat[i,j]+mu*(VMat[i,j]-pre_v)
                if(ACMat[i,j]<=dlt):
                    ACMat[i,j]=2*dlt
                    VMat[i,j]=-VMat[i,j]
                elif(ACMat[i,j]>1.0-dlt):
                    ACMat[i,j]=1.0-dlt
                    VMat[i,j]=-VMat[i,j]
    #
    return(CLL_Rfu,Err,ACMat,VMat)
################################################################################
def Err_Plot(Name,X_l,Y_Err,Y_lt1,Y_lt2,Y_lt3,Y_lt4,Y_lt5,Y_lt6):
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
    mpl.plot(X_l,Y_lt4,'ro',markersize=2,label='Last lt4='+str(Y_lt4[-1]))
    mpl.plot(X_l,Y_lt5,'bo',markersize=2,label='Last lt5='+str(Y_lt5[-1]))
    mpl.plot(X_l,Y_lt6,'go',markersize=2,label='Last lt6='+str(Y_lt6[-1]))
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
    #
    GasT=300.0
    WallT=300.0
    GasTGasmpV=math.sqrt(2*1.38E-23*GasT/(39.95/6.02*1E-26))
    WallTGasmpV=math.sqrt(2*1.38E-23*WallT/(39.95/6.02*1E-26))
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
    SamplesN=int(N/10)
    SamplesUAN,SampleID,SampleVX,SampleVY,SampleVZ,SampleFVX,SampleFVY,SampleFVZ,SampleAds=Random_Sampling(N,SamplesN,ID,VX,VY,VZ,FVX,FVY,FVZ,Adsorbed)
    #
    I=50
    #
    cri_Time,Time_Count,cri_f=Criterion(N,UnA_N,Tt,Adsorbed,200,35,0.0,3500.0)
    Criterion_Plot('Interaction Time',cri_Time,cri_f,Time_Count,35,0.0,0.07,0.0,7000.0)
    #
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
    #Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir)
    Swi,Swr,SRfwir=Distribution_Cloud(SamplesN,SamplesUAN,SampleVZ,SampleFVZ,SampleAds,I,I,-3.0,0.0,0.0,3.0)
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Sample',0,SRfwir)
    wi,wr,Rfwir,EwnExcMat,EwtExcMat,EwTtExcMat,ERwntInMat,ERwntOutMat,ERwntExcMat=Distribution_Cloud_Energy(N,UnA_N,VZ,FVZ,Adsorbed,I,I,-3.0,0.0,0.0,3.0,VX,VY,FVX,FVY)
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Energy',0,EwnExcMat,'EEn')
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Energy',0,EwtExcMat,'EEt')
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Energy',0,EwTtExcMat,'EETt')
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Energy',0,ERwntInMat,'ERntIn')
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Energy',0,ERwntOutMat,'ERntOut')
    Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Energy',0,ERwntExcMat,'ERntExc')
    #Cloud_dat('VZ','FVZ',I,I,wi,wr,Rfwir,'Energy',0,EwvExcMat,'EEY')
    #
    u,fui,fur,fua,fuUa=Distribution(N,UnA_N,VX,FVX,Adsorbed,I,-3.0,3.0)
    v,fvi,fvr,fva,fvUa=Distribution(N,UnA_N,VY,FVY,Adsorbed,I,-3.0,3.0)
    #Distribution_Plot('VY',v,fvi,fvr,fva,-3.0,3.0,0.0,1.0)
    w,fwi,fwr,fwa,fwUa=Distribution(N,UnA_N,VZ,FVZ,Adsorbed,2*I,-3.0,3.0)
    #Distribution_Plot('VZ',w,fwi,fwr,fwa,-3.0,3.0,0.0,1.4)
    #
    Loops=10000
    mu=0.999
    lt1_x=1.45
    lt2_x=0.88
    lt3_x=0.88
    v1_x=0.0
    v2_x=0.0
    v3_x=0.0
    lt1_y=1.45
    lt2_y=0.88
    lt3_y=0.88
    v1_y=0.0
    v2_y=0.0
    v3_y=0.0
    lt1_z=0.3
    lt2_z=0.7
    lt3_z=0.5
    lt4_z=0.5
    lt5_z=0.33
    lt6_z=0.5
    v1_z=0.0
    v2_z=0.0
    v3_z=0.0
    v4_z=0.0
    v5_z=0.0
    v6_z=0.0
    Loop=np.zeros(Loops+1)
    Err_X=np.zeros(Loops+1)
    Err_Y=np.zeros(Loops+1)
    Err_Z=np.zeros(Loops+1)
    lt1_X=np.zeros(Loops+1)
    lt2_X=np.zeros(Loops+1)
    lt3_X=np.zeros(Loops+1)
    lt1_Y=np.zeros(Loops+1)
    lt2_Y=np.zeros(Loops+1)
    lt3_Y=np.zeros(Loops+1)
    lt1_Z=np.zeros(Loops+1)
    lt2_Z=np.zeros(Loops+1)
    lt3_Z=np.zeros(Loops+1)
    lt4_Z=np.zeros(Loops+1)
    lt5_Z=np.zeros(Loops+1)
    lt6_Z=np.zeros(Loops+1)
    ACZMat=np.ones((len(wi),len(wr)))*0.5
    VZMat=np.zeros((len(wi),len(wr)))
    with open('Fitting.log','w') as out:
        print('Loop\tErr_X\tlt1_X\tlt2_X\tlt3_X\tErr_Y\tlt1_Y\tlt2_Y\tlt3_Y\tErr_Z\tlt1_Z\tlt2_Z\tlt3_Z\tlt4_Z\tlt5_Z\tlt6_Z',file=out)
    #
    for l in range(Loops+1):
        #CLL_Rfu,CLL_Noru,CLL_fu,CLL_fui,CLL_fur,Errx,lt1_x,lt2_x,lt3_x,v1_x,v2_x,v3_x=Gradient_Least_Squares(0,N,ui,ur,Rfuir,fui,VX,FVX,VY,VZ,GasTGasmpV,WallTGasmpV,lt1_x,lt2_x,lt3_x,mu,v1_x,v2_x,v3_x)
        #CLL_Rfv,CLL_Norv,CLL_fv,CLL_fvi,CLL_fvr,Erry,lt1_y,lt2_y,lt3_y,v1_y,v2_y,v3_y=Gradient_Least_Squares(0,N,vi,vr,Rfvir,fvi,VY,FVY,VX,VZ,GasTGasmpV,WallTGasmpV,lt1_y,lt2_y,lt3_y,mu,v1_y,v2_y,v3_y)
        #CLL_Rfw,CLL_Norw,CLL_fw,CLL_fwi,CLL_fwr,Errz,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z=Gradient_Least_Squares(1,N,wi,ui,vi,wr,Rfwir,fwUa,fuUa,fvUa,VZ,FVZ,VX,VY,Adsorbed,GasTGasmpV,WallTGasmpV,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,mu,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z)
        #Eps_Rfw,Eps_Norw,Eps_fw,Eps_fwi,Eps_fwr,Errz,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z=Gradient_Least_Squares(Epstein,1,N,wi,ui,vi,wr,Rfwir,fwUa,fuUa,fvUa,VZ,FVZ,VX,VY,Adsorbed,WallTGasmpV,WallTGasmpV,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,mu,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z)
        Rec_Rfw,Rec_Norw,Rec_Fw,Rec_fwi,Rec_fwr,Errz,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z=Gradient_Least_Squares(Reciprocal,1,N,wi,ui,vi,wr,Rfwir,fwUa,fuUa,fvUa,VZ,FVZ,VX,VY,Adsorbed,WallTGasmpV,WallTGasmpV,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,mu,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z)
        #CLL_Rfw,ErrACZ,ACZMat,VZMat=AC(1,wi,wr,Rfwir,fwUa,GasTGasmpV,WallTGasmpV,ACZMat,VZMat)
        #CLL_fwi=np.zeros(len(fwi))
        #CLL_fwr=np.zeros(len(fwr))
        #CLL_Norw=np.zeros(len(fwi))
        print(l,Errz,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,Rec_Fw)
        #print(l,ErrACZ)
        with open('Fitting.log','a') as out:
            print('%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f'%(l,0.0,lt1_x,lt2_x,lt3_x,0.0,lt1_y,lt2_y,lt3_y,Errz,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z),file=out)
        Loop[l]=l
        #Err_X[l]=Errx
        #lt1_X[l]=lt1_x
        #lt2_X[l]=lt2_x
        #lt3_X[l]=lt3_x
        #Err_Y[l]=Erry
        #lt1_Y[l]=lt1_y
        #lt2_Y[l]=lt2_y
        #lt3_Y[l]=lt3_y
        Err_Z[l]=Errz
        lt1_Z[l]=lt1_z
        lt2_Z[l]=lt2_z
        lt3_Z[l]=lt3_z
        lt4_Z[l]=lt4_z
        lt5_Z[l]=lt5_z
        lt6_Z[l]=lt6_z
        if(l%100==0):
            #Cloud_dat('VX','CLL_FVX',I,I,ui,ur,CLL_Rfu,'CLL',l,Rfuir)
            #Cloud_dat('VY','CLL_FVY',I,I,vi,vr,CLL_Rfv,'CLL',l,Rfvir)
            Cloud_dat('VZ','CLL_FVZ',I,I,wi,wr,Rec_Rfw,'CLL',l,Rfwir)
            #Cloud_dat('VZ','FVZ',I,I,wi,wr,ACZMat,'AC',l,Rfwir)
    #
    #Err_Plot('X',Loop,Err_X,lt1_X,lt2_X,lt3_X)
    #Err_Plot('Y',Loop,Err_Y,lt1_Y,lt2_Y,lt3_Y)
    Err_Plot('Z',Loop,Err_Z,lt1_Z,lt2_Z,lt3_Z,lt4_Z,lt5_Z,lt6_Z)
    #Distribution_Plot('VX',u,fui,CLL_fui,fur,CLL_fur,fua,CLL_Noru,-3.0,3.0,0.0,1.0)
    #Distribution_Plot('VY',v,fvi,CLL_fvi,fvr,CLL_fvr,fva,CLL_Norv,-3.0,3.0,0.0,1.0)
    Distribution_Plot('VZ',w,fwi,fwr,fwa,fwUa,Rec_fwi,Rec_fwr,Rec_Norw,-3.0,3.0,0.0,1.6,GasT)
################################################################################
if __name__ == '__main__':
    #
    main()
