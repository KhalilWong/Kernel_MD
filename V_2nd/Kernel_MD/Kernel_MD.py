import time
import numpy as np
import matplotlib.pyplot as mpl

################################################################################
class Parameters():

    def __init__(self):
        #物理参数
        self.pi=3.14159265
        self.Mass=np.array([39.95,195.08])/6.02*1E-26#单位kg
        self.LJ_E=np.array([1.654E-21,5.207E-20,1.093E-21])#单位J
        self.LJ_S=np.array([3.40,2.47,2.94])*1E-10#单位m
        self.fcc_lattice=3.93E-10
        self.kB=1.38E-23
        self.T=np.array([300.,300.])
        self.mp_V=np.array([np.sqrt(2*self.kB*self.T[0]/self.Mass[0]),np.sqrt(3*self.kB*self.T[1]/self.Mass[1])])#气体最概然速率，固体方均根速率
        #无量纲参数
        self.nd_Mass=self.Mass[1]
        self.nd_Energy=self.LJ_E[1]
        self.nd_Length=self.LJ_S[1]
        self.nd_Velocity=np.sqrt(self.nd_Energy/self.nd_Mass)
        self.nd_Time=self.nd_Length/self.nd_Velocity
        self.nd_Acceleration=self.nd_Energy/(self.nd_Mass*self.nd_Length)
        #无量纲化
        self.Mass/=self.nd_Mass
        self.LJ_E/=self.nd_Energy
        self.LJ_S/=self.nd_Length
        self.cutoff=10*1E-10/self.nd_Length
        self.fcc_lattice/=self.nd_Length
        self.mp_V/=self.nd_Velocity
        self.d=5.0
        self.spr_k=5000.
        self.dt=0.001
        self.Tt=3500000
        #盒子参数
        self.Pt_I=6
        self.Pt_J=6
        self.Pt_K=3
        self.Box=np.zeros((3,3))
        self.Pt_ePos=np.zeros((4*self.Pt_I*self.Pt_J*self.Pt_K,3))
        self.Pt_N=0
        self.Pt_type=np.ones(4*self.Pt_I*self.Pt_J*self.Pt_K)
        self.Ar_N=0
        self.Ar_type=np.zeros(1)
        #状态参数
        self.state=True
        self.dumpstep=100

################################################################################
def Initialization(Pars):
    timestep=0
    #初始化Pt的初始位置和速度
    Pt_Pos=np.zeros((4*Pars.Pt_I*Pars.Pt_J*Pars.Pt_K,3))
    Pt_Vel=np.zeros((4*Pars.Pt_I*Pars.Pt_J*Pars.Pt_K,3))
    count=0
    for i in range(2*Pars.Pt_I):
        for j in range(2*Pars.Pt_J):
            for k in range(2*Pars.Pt_K):
                if (i/2+j/2+k/2==int(i/2+j/2+k/2)):
                    count+=1
                    Pt_Pos[count-1,0]=i/2*Pars.fcc_lattice
                    Pt_Pos[count-1,1]=j/2*Pars.fcc_lattice
                    Pt_Pos[count-1,2]=(k/2-2.5)*Pars.fcc_lattice
                    for dim in range(3):
                        R1=0
                        while R1==0:
                            R1=np.random.rand()
                        R2=0
                        while R2==0:
                            R2=np.random.rand()
                        Pt_Vel[count-1,dim]=Pars.mp_V[1]/np.sqrt(3)*np.sqrt(-2*np.log(R1))*np.cos(2*Pars.pi*R2)#高斯分布，平均值为0，方差=方均根速度用来控温
    Pars.Pt_N=count
    Pars.Box[0,0]=np.min(Pt_Pos[:,0])
    Pars.Box[0,1]=np.max(Pt_Pos[:,0])+Pars.fcc_lattice/2
    Pars.Box[0,2]=Pars.Box[0,1]-Pars.Box[0,0]
    Pars.Box[1,0]=np.min(Pt_Pos[:,1])
    Pars.Box[1,1]=np.max(Pt_Pos[:,1])+Pars.fcc_lattice/2
    Pars.Box[1,2]=Pars.Box[1,1]-Pars.Box[1,0]
    Pars.Box[2,0]=np.min(Pt_Pos[:,2])
    Pars.Box[2,1]=Pars.d
    Pars.Box[2,2]=Pars.Box[2,1]-Pars.Box[2,0]
    print('计算区域X:',Pars.Box[0,0],',',Pars.Box[0,1])
    print('计算区域Y:',Pars.Box[1,0],',',Pars.Box[1,1])
    print('计算区域Z:',Pars.Box[2,0],',',Pars.Box[2,1])
    #初始化Ar的初始位置和速度，无量纲化
    Ar_Pos=np.zeros((1,3))
    Ar_Vel=np.zeros((1,3))
    Pars.Ar_N=1
    Ar_Pos[0,0]=Pars.Box[0,0]+Pars.Box[0,2]*np.random.random()
    Ar_Pos[0,1]=Pars.Box[1,0]+Pars.Box[1,2]*np.random.random()
    Ar_Pos[0,2]=Pars.Box[2,1]
    R1=0
    while R1==0:
        R1=np.random.rand()
    R2=0
    while R2==0:
        R2=np.random.rand()
    Ar_Vel[0,0]=Pars.mp_V[0]*np.sqrt(-np.log(R1))*np.cos(2*Pars.pi*R2)#Maxwell分布
    R1=0
    while R1==0:
        R1=np.random.rand()
    R2=0
    while R2==0:
        R2=np.random.rand()
    Ar_Vel[0,1]=Pars.mp_V[0]*np.sqrt(-np.log(R1))*np.sin(2*Pars.pi*R2)
    R1=0
    while R1==0:
        R1=np.random.rand()
    R2=0
    while R2==0:
        R2=np.random.rand()
    Ar_Vel[0,2]=-Pars.mp_V[0]*np.sqrt(-np.log(R1))
    #初始信息
    print('Created ',Pars.Pt_N,' Pt')
    print('Created ',Pars.Ar_N,' Ar')
    Pt_V2=0
    for i in range(Pars.Pt_N):
        Pt_V2+=Pt_Vel[i,0]**2+Pt_Vel[i,1]**2+Pt_Vel[i,2]**2
    Pt_T=Pt_V2*Pars.nd_Velocity**2*Pars.Mass[1]*Pars.nd_Mass/(3*Pars.Pt_N*Pars.kB)
    print('Pt整体x方向平均速度',np.sum(Pt_Vel[:,0])/Pars.Pt_N)
    print('Pt整体y方向平均速度',np.sum(Pt_Vel[:,1])/Pars.Pt_N)
    print('Pt整体z方向平均速度',np.sum(Pt_Vel[:,2])/Pars.Pt_N)
    print('Pt温度',Pt_T)
    print('Ar入射速度:',Ar_Vel[0,0],',',Ar_Vel[0,1],',',Ar_Vel[0,2])
    print('*******Model Initialization Done!*******')
    #首次控温
    Pt_Vel=rescale_T(Pt_Vel,Pars)
    Pars.Pt_ePos=Pt_Pos
    #首次位置周期
    (Ar_Pos,Pt_Pos)=Pos_period(Ar_Pos,Pt_Pos,Pars)
    #首次加速度周期
    (Ar_Acc,Pt_Acc)=Acceleration_period(Ar_Pos,Pt_Pos,Pars)

    return(Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc,timestep)


################################################################################
def Pos_period(Ar_Pos,Pt_Pos,Pars):
    #X,Y方向周期
    for axis in range(2):
        #Pt
        for i in range(Pars.Pt_N):
            if(Pt_Pos[i,axis]<Pars.Box[axis,0]):
                Pt_Pos[i,axis]+=Pars.Box[axis,2]
                Pars.Pt_ePos[i,axis]+=Pars.Box[axis,2]
            elif(Pt_Pos[i,axis]>=Pars.Box[axis,1]):
                Pt_Pos[i,axis]-=Pars.Box[axis,2]
                Pars.Pt_ePos[i,axis]-=Pars.Box[axis,2]
        #Ar
        if(Ar_Pos[0,axis]<Pars.Box[axis,0]):
            Ar_Pos[0,axis]+=Pars.Box[axis,2]
        elif(Ar_Pos[0,axis]>=Pars.Box[axis,1]):
            Ar_Pos[0,axis]-=Pars.Box[axis,2]
    #Z方向下边界更新
    Pars.Box[2,0]=np.min(Pt_Pos[:,2])
    Pars.Box[2,2]=Pars.Box[2,1]-Pars.Box[2,0]

    return(Ar_Pos,Pt_Pos)


################################################################################
def Acceleration_period(Ar_Pos,Pt_Pos,Pars):
    All_Pos=np.vstack((Pt_Pos,Ar_Pos))
    All_type=np.hstack((Pars.Pt_type,Pars.Ar_type))
    All_N=Pars.Pt_N+Pars.Ar_N
    Pt_F=np.zeros((Pars.Pt_N,3))
    for i in range(All_N):
        Atom_F=np.zeros((1,3))
        for j in range(All_N):
            if(All_type[i]==1 and All_type[j]==1):
                LJ_pair=1
            elif(All_type[i]==0 and All_type[j]==0):
                LJ_pair=0
            else:
                LJ_pair=2
            Epair=Pars.LJ_E[LJ_pair]
            Spair=Pars.LJ_S[LJ_pair]
            #周期相对位置
            Pair=All_Pos[i,:]-All_Pos[j,:]
            for axis in range(2):
                if(abs(Pair[axis])>=Pars.Box[axis,2]-Pars.cutoff):
                    Pair[axis]=Pair[axis]-Pars.Box[axis,2]*Pair[axis]/abs(Pair[axis])
            #周期距离
            Dispair=np.sqrt(Pair[0]**2+Pair[1]**2+Pair[2]**2)
            if(0<Dispair<=Pars.cutoff):
                Fpair=48*Epair*(Spair**12/Dispair**13-0.5*Spair**6/Dispair**7)
                Atom_F+=Pair*Fpair/Dispair
        if(All_type[i]==1):
            Pt_F[i,:]=Atom_F
        else:
            Ar_F=Atom_F
    #Pt弹性恢复力
    Spring_Dis=All_Pos[:Pars.Pt_N,:]-Pars.Pt_ePos
    Spring_F=-Pars.spr_k*Spring_Dis
    Pt_F+=Spring_F
    Pt_Acc=Pt_F/Pars.Mass[1]
    Ar_Acc=Ar_F/Pars.Mass[0]

    return(Ar_Acc,Pt_Acc)


################################################################################
def Verlet(Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc,Pars):
    Ar_Pos_n=Ar_Pos+Ar_Vel*Pars.dt+0.5*Ar_Acc*Pars.dt**2
    Pt_Pos_n=Pt_Pos+Pt_Vel*Pars.dt+0.5*Pt_Acc*Pars.dt**2
    (Ar_Pos_n,Pt_Pos_n)=Pos_period(Ar_Pos_n,Pt_Pos_n,Pars)
    (Ar_Acc_n,Pt_Acc_n)=Acceleration_period(Ar_Pos_n,Pt_Pos_n,Pars)
    Ar_Vel_n=Ar_Vel+0.5*(Ar_Acc+Ar_Acc_n)*Pars.dt
    Pt_Vel_n=Pt_Vel+0.5*(Pt_Acc+Pt_Acc_n)*Pars.dt

    return(Ar_Pos_n,Ar_Vel_n,Ar_Acc_n,Pt_Pos_n,Pt_Vel_n,Pt_Acc_n)


################################################################################
def rescale_T(Pt_Vel,Pars):
    Pt_argVel=np.array([np.sum(Pt_Vel[:,0])/Pars.Pt_N,np.sum(Pt_Vel[:,1])/Pars.Pt_N,np.sum(Pt_Vel[:,2])/Pars.Pt_N])#只需要热运动速度
    Pt_Vel[:,0]-=Pt_argVel[0]
    Pt_Vel[:,1]-=Pt_argVel[1]
    Pt_Vel[:,2]-=Pt_argVel[2]
    Pt_V2=0
    for i in range(Pars.Pt_N):
        Pt_V2+=Pt_Vel[i,0]**2+Pt_Vel[i,1]**2+Pt_Vel[i,2]**2
    Pt_T=Pt_V2*Pars.nd_Velocity**2*Pars.Mass[1]*Pars.nd_Mass/(3*Pars.Pt_N*Pars.kB)
    Pt_Vel*=np.sqrt(Pars.T[1]/Pt_T)

    return(Pt_Vel)


################################################################################
def Dump(Pt_Pos,Ar_Pos,Ar_Z,Ar_t,Pars,timestep,dumpstep=1):
    if(timestep%dumpstep==0):
        with open('Kernel_MD.dump','a') as MD:
            print('ITEM: TIMESTEP',file=MD)
            print(timestep,file=MD)
            print('ITEM: NUMBER OF ATOMS',file=MD)
            print(Pars.Pt_N+Pars.Ar_N,file=MD)
            print('ITEM: BOX BOUNDS pp pp ff',file=MD)
            print(Pars.Box[0,0],Pars.Box[0,1],file=MD)
            print(Pars.Box[1,0],Pars.Box[1,1],file=MD)
            print(Pars.Box[2,0],Pars.Box[2,1],file=MD)
            print('ITEM: ATOMS id type x y z',file=MD)
            for i in range(Pars.Pt_N):
                print(i+1,int(Pars.Pt_type[i])+1,Pt_Pos[i,0],Pt_Pos[i,1],Pt_Pos[i,2],file=MD)
            print(Pars.Pt_N+1,int(Pars.Ar_type[0])+1,Ar_Pos[0,0],Ar_Pos[0,1],Ar_Pos[0,2],file=MD)

        if(Pars.state):
            Ar_Z.append(Ar_Pos[0,2])
            Ar_t.append(timestep*Pars.dt)

    return(Ar_Z,Ar_t)


################################################################################
def Exit(Pt_Pos,Ar_Pos,Ar_Z,Ar_t,Pars,timestep):
    if(Ar_Pos[0,2]>Pars.d or timestep>=Pars.Tt):
        Pars.state=False
        (Ar_Z,Ar_t)=Dump(Pt_Pos,Ar_Pos,Ar_Z,Ar_t,Pars,timestep)
    else:
        (Ar_Z,Ar_t)=Dump(Pt_Pos,Ar_Pos,Ar_Z,Ar_t,Pars,timestep,Pars.dumpstep)


################################################################################
#***********************************Others*************************************#
################################################################################

################################################################################
def Fplot(Pars,Pt_Vel,Ar_Z,Ar_t,timestep,argtime):
    nacc=25
    mp_V2=[]
    for i in range(Pars.Pt_N):
        mp_V2.append(Pt_Vel[i,0]**2+Pt_Vel[i,1]**2+Pt_Vel[i,2]**2)
    mp_V=[np.sqrt(i) for i in mp_V2]
    print('x方向平均速度'+str(np.sum(Pt_Vel[:,0])/Pars.Pt_N))
    print('y方向平均速度'+str(np.sum(Pt_Vel[:,1])/Pars.Pt_N))
    print('z方向平均速度'+str(np.sum(Pt_Vel[:,2])/Pars.Pt_N))
    print('温度'+str(np.sum(mp_V2)*Pars.nd_Velocity**2*Pars.Mass[1]*Pars.nd_Mass/(3*Pars.Pt_N*Pars.kB)))
    test=[int(round(i*nacc)) for i in mp_V]
    maxu=np.max(test)
    minu=np.min(test)
    print(maxu/nacc,minu/nacc)
    fu=np.zeros((maxu-minu+1,2))
    thex=[x/nacc/10 for x in range(minu*10,(maxu+1)*10)]
    thefmax=[4*Pars.pi*(3/(2*Pars.pi*Pars.mp_V[1]**2))**(3/2)*np.exp(-3*thexi**2/(2*Pars.mp_V[1]**2))*thexi**2 for thexi in thex]
    du=minu
    while minu<=du<=maxu:
        fu[du-minu,0]=du/nacc
        fu[du-minu,1]=test.count(du)*nacc/Pars.Pt_N
        du+=1
    print(np.sum(fu[:,1])/nacc)
    #图1
    fig1=mpl.figure()
    mpl.plot(fu[:,0],fu[:,1],'ro',markersize=1,label='Random')
    mpx=[Pars.mp_V[1] for x in range(250)]
    mpy=[y/100 for y in range(250)]
    mpl.plot(mpx,mpy,'g--',markersize=1,label='RMS')
    mpl.plot(thex,thefmax,'b',markersize=1,label='Maxwell')
    mpl.legend(loc='upper left')
    mpl.xlabel('$v^{*}$')
    mpl.ylabel('$f(v^{*})$')
    mpl.savefig('Pt_T_Distribution.png',dpi=600)
    #图2
    fig2=mpl.figure()
    eff=str(timestep)+' TimeSteps; ArgTime: '+str(round(argtime,10))+' Seconds;'
    mpl.plot(Ar_t,Ar_Z,'ro',markersize=1,label=eff)
    mpl.legend(loc='upper center')
    mpl.xlabel('$t$')
    mpl.ylabel('$Z$')
    mpl.savefig('Ar_Z_t.png',dpi=600)

    mpl.show()

################################################################################
def Bar_show(Pars,current,start_time,Name):
    bar_length=10
    pre=' |'
    suf='| '
    fill='◉'
    emp='◯'

    per=int(current*100/Pars.Tt)
    n=int(current*bar_length/Pars.Tt)
    rn=int(bar_length-n)
    step_time=time.time()
    argtime=(step_time-start_time)/current
    show_string=Name+': '+pre+fill*n+emp*rn+suf+str(current)+' / '+str(Pars.Tt)+' ArgTime: '+str(round(argtime,10))+' Seconds'
    if(current!=Pars.Tt):
        print(show_string, end='\r', flush=True)
        #time.sleep(0.01)

    return(argtime)

################################################################################
def Bar_close(start_time,Name):
    end_time=time.time()
    alltime=end_time-start_time
    close_string='\n'+'*******'+Name+' '+'Done! '+'AllTime: '+str(round(alltime,10))+' Seconds!'+'*******'
    print(close_string)
    #time.sleep(1)

################################################################################
#*************************************Main*************************************#
################################################################################

Pars=Parameters()
(Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc,timestep)=Initialization(Pars)
Ar_Z=[]
Ar_t=[]
Dump(Pt_Pos,Ar_Pos,Ar_Z,Ar_t,Pars,timestep)
start_time=time.time()
while(Pars.state):
    (Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc)=Verlet(Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc,Pars)
    (Pt_Vel)=rescale_T(Pt_Vel,Pars)
    timestep+=1
    Exit(Pt_Pos,Ar_Pos,Ar_Z,Ar_t,Pars,timestep)
    argtime=Bar_show(Pars,timestep,start_time,'Kernel_MD')
Bar_close(start_time,'Kernel_MD')
Fplot(Pars,Pt_Vel,Ar_Z,Ar_t,timestep,argtime)
