from numpy import *
from matplotlib.pyplot import *
################################################################################
def Initialization(Pars):
    timestep=0
    Pt_I=6
    Pt_J=6
    Pt_K=3
    #初始化Pt的初始位置和速度
    Pt_Pos=empty(shape=[0,3])
    Pt_Vel=empty(shape=[0,3])
    count=0
    for i in range(2*Pt_I):
        for j in range(2*Pt_J):
            for k in range(2*Pt_K):
                if (i/2+j/2+k/2==int(i/2+j/2+k/2)):
                    count+=1
                    Pars.Pt_type.append(1)
                    Pt_Pos=append(Pt_Pos,[[i/2*Pars.fcc_lattice,j/2*Pars.fcc_lattice,(k/2-2.5)*Pars.fcc_lattice]],axis=0)
                    Pt_uvw=[]
                    for dim in range(3):
                        R1=0
                        while R1==0:
                            R1=random.rand()
                        R2=0
                        while R2==0:
                            R2=random.rand()
                        Pt_uvw.append(Pars.mp_V[1]/sqrt(3)*sqrt(-2*log(R1))*cos(2*Pars.pi*R2))#高斯分布，平均值为0，方差=方均根速度用来控温
                    Pt_Vel=append(Pt_Vel,[Pt_uvw],axis=0)
    Pars.Pt_N=count
    print('Created '+str(Pars.Pt_N)+' Pt')
    #初始化Ar的初始位置和速度，无量纲化
    Pars.Box[0,0]=min(Pt_Pos[:,0])
    Pars.Box[0,1]=max(Pt_Pos[:,0])+Pars.fcc_lattice/2
    Pars.Box[0,2]=Pars.Box[0,1]-Pars.Box[0,0]
    Pars.Box[1,0]=min(Pt_Pos[:,1])
    Pars.Box[1,1]=max(Pt_Pos[:,1])+Pars.fcc_lattice/2
    Pars.Box[1,2]=Pars.Box[1,1]-Pars.Box[1,0]
    Pars.Box[2,0]=min(Pt_Pos[:,2])
    Pars.Box[2,1]=Pars.d
    Pars.Box[2,2]=Pars.Box[2,1]-Pars.Box[2,0]
    print('计算区域X:'+str(Pars.Box[0,0])+','+str(Pars.Box[0,1]))
    print('计算区域Y:'+str(Pars.Box[1,0])+','+str(Pars.Box[1,1]))
    print('计算区域Z:'+str(Pars.Box[2,0])+','+str(Pars.Box[2,1]))
    Ar_Pos=zeros(3)
    Ar_Vel=zeros(3)
    Pars.Ar_type=0
    Pars.Ar_N=1
    Ar_Pos[0]=(Pars.Box[0,0]+Pars.Box[0,1])/2
    Ar_Pos[1]=(Pars.Box[1,0]+Pars.Box[1,1])/2
    Ar_Pos[2]=Pars.Box[2,1]
    R1=0
    while R1==0:
        R1=random.rand()
    R2=0
    while R2==0:
        R2=random.rand()
    Ar_Vel[0]=Pars.mp_V[0]*sqrt(-log(R1))*cos(2*Pars.pi*R2)#Maxwell分布
    R1=0
    while R1==0:
        R1=random.rand()
    R2=0
    while R2==0:
        R2=random.rand()
    Ar_Vel[1]=Pars.mp_V[0]*sqrt(-log(R1))*sin(2*Pars.pi*R2)
    R1=0
    while R1==0:
        R1=random.rand()
    R2=0
    while R2==0:
        R2=random.rand()
    Ar_Vel[2]=-Pars.mp_V[0]*sqrt(-log(R1))
    #首次控温
    Pt_Vel=rescale_T(Pt_Vel,Pars)
    Pars.Pt_ePos=Pt_Pos
    #首次周期处理
    (Ar_Pos,Pt_Pos,Unreal_Pos,Unreal_type,Unreal_N)=Pos_period(Ar_Pos,Pt_Pos,Pars)
    #首次加速度
    (Ar_Acc,Pt_Acc)=Acceleration(Unreal_Pos,Unreal_type,Unreal_N,Pars)
    print('Ar入射速度:'+str(Ar_Vel[0])+','+str(Ar_Vel[1])+','+str(Ar_Vel[2]))
    print('*******Model Initialization Done!*******')

    return(Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc,timestep)


################################################################################
def Acceleration_old(Unreal_Pos,Unreal_type,Unreal_N,Pars):
    Pt_F=empty(shape=[0,3])
    for i in range(Pars.Pt_N+Pars.Ar_N):
        LJ_pair=[]
        for j in range(Unreal_N):
            if(Unreal_type[i]==1 and Unreal_type[j]==1):
                LJ_pair.append(1)
            elif(Unreal_type[i]==0 and Unreal_type[j]==0):
                LJ_pair.append(0)
            else:
                LJ_pair.append(2)
        Epair=[Pars.LJ_E[i] for i in LJ_pair]
        Spair=[Pars.LJ_S[i] for i in LJ_pair]
        Dis3D=tile(Unreal_Pos[i,:],(Unreal_N,1))-Unreal_Pos
        Dispair=[sqrt(pair[0]**2+pair[1]**2+pair[2]**2) for pair in Dis3D]
        Fpair=[]
        for j in range(Unreal_N):
            if(0<Dispair[j]<=Pars.cutoff):
                Fpair.append((48*Epair[j]*(Spair[j]**12/Dispair[j]**13-0.5*Spair[j]**6/Dispair[j]**7))/Dispair[j])
            elif(Dispair[j]==0):
                Fpair.append(0.0)
            else:
                delete(Dis3D,j,axis=0)
        F_N=size(Fpair)
        Atom_F=zeros(3)
        for j in range(F_N):
            Atom_F+=Dis3D[j,:]*Fpair[j]
        if(Unreal_type[i]==1):
            Pt_F=append(Pt_F,[Atom_F],axis=0)
        else:
            Ar_F=Atom_F
    #Pt弹性恢复力
    Spring_Dis=Unreal_Pos[:Pars.Pt_N,:]-Pars.Pt_ePos
    Spring_F=-Pars.spr_k*Spring_Dis
    Pt_F+=Spring_F
    Pt_Acc_n=Pt_F/Pars.Mass[1]
    Ar_Acc_n=Ar_F/Pars.Mass[0]

    return(Ar_Acc_n,Pt_Acc_n)


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
def Pos_period_old(Ar_Pos,Pt_Pos,Pars):
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
        if(Ar_Pos[axis]<Pars.Box[axis,0]):
            Ar_Pos[axis]+=Pars.Box[axis,2]
        elif(Ar_Pos[axis]>=Pars.Box[axis,1]):
            Ar_Pos[axis]-=Pars.Box[axis,2]
    #Z方向下边界更新
    Pars.Box[2,0]=min(Pt_Pos[:,2])
    Pars.Box[2,2]=Pars.Box[2,1]-Pars.Box[2,0]
    #周期虚拟原子
    Unreal_Pos=Pt_Pos
    Unreal_Pos=append(Unreal_Pos,[Ar_Pos],axis=0)
    Unreal_type=Pars.Pt_type
    Unreal_type=append(Unreal_type,[Pars.Ar_type])
    Unreal_N=Pars.Pt_N+Pars.Ar_N
    Add_N=0
    #X方向
    for i in range(Unreal_N):
        if(Unreal_Pos[i,0]<=Pars.Box[0,0]+Pars.cutoff):
            Add_N+=1
            Unreal_Pos=append(Unreal_Pos,[[Unreal_Pos[i,0]+Pars.Box[0,2],Unreal_Pos[i,1],Unreal_Pos[i,2]]],axis=0)
            Unreal_type=append(Unreal_type,[Unreal_type[i]])
        if(Unreal_Pos[i,0]>=Pars.Box[0,1]-Pars.cutoff):
            Add_N+=1
            Unreal_Pos=append(Unreal_Pos,[[Unreal_Pos[i,0]-Pars.Box[0,2],Unreal_Pos[i,1],Unreal_Pos[i,2]]],axis=0)
            Unreal_type=append(Unreal_type,[Unreal_type[i]])
    Unreal_N+=Add_N
    Add_N=0
    #Y方向
    for i in range(Unreal_N):
        if(Unreal_Pos[i,1]<=Pars.Box[1,0]+Pars.cutoff):
            Add_N+=1
            Unreal_Pos=append(Unreal_Pos,[[Unreal_Pos[i,0],Unreal_Pos[i,1]+Pars.Box[1,2],Unreal_Pos[i,2]]],axis=0)
            Unreal_type=append(Unreal_type,[Unreal_type[i]])
        if(Unreal_Pos[i,1]>=Pars.Box[1,1]-Pars.cutoff):
            Add_N+=1
            Unreal_Pos=append(Unreal_Pos,[[Unreal_Pos[i,0],Unreal_Pos[i,1]-Pars.Box[1,2],Unreal_Pos[i,2]]],axis=0)
            Unreal_type=append(Unreal_type,[Unreal_type[i]])
    Unreal_N+=Add_N

    return(Ar_Pos,Pt_Pos,Unreal_Pos,Unreal_type,Unreal_N)

################################################################################
def rescale_T(Pt_Vel,Pars):
    Pt_argVel=[sum(Pt_Vel[:,0])/Pars.Pt_N,sum(Pt_Vel[:,1])/Pars.Pt_N,sum(Pt_Vel[:,2])/Pars.Pt_N]#只需要热运动速度
    Pt_Vel[:,0]-=Pt_argVel[0]
    Pt_Vel[:,1]-=Pt_argVel[1]
    Pt_Vel[:,2]-=Pt_argVel[2]
    Pt_V2=0
    for i in range(Pars.Pt_N):
        Pt_V2+=Pt_Vel[i,0]**2+Pt_Vel[i,1]**2+Pt_Vel[i,2]**2
    Pt_T=Pt_V2*Pars.nd_Velocity**2*Pars.Mass[1]*Pars.nd_Mass/(3*Pars.Pt_N*Pars.kB)
    Pt_Vel*=sqrt(Pars.T[1]/Pt_T)
    Pt_V2=0
    for i in range(Pars.Pt_N):
        Pt_V2+=Pt_Vel[i,0]**2+Pt_Vel[i,1]**2+Pt_Vel[i,2]**2
    Pt_T=Pt_V2*Pars.nd_Velocity**2*Pars.Mass[1]*Pars.nd_Mass/(3*Pars.Pt_N*Pars.kB)
    #print('Pt温度'+str(Pt_T))

    return(Pt_Vel)


################################################################################
def Dump(Pt_Pos,Ar_Pos,Pars,timestep,dumpstep=1):
    if(timestep%dumpstep==0):
        with open('MD.dump','a') as MD:
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
                print(i+1,Pars.Pt_type[i]+1,Pt_Pos[i,0],Pt_Pos[i,1],Pt_Pos[i,2],file=MD)
            print(Pars.Pt_N+1,Pars.Ar_type+1,Ar_Pos[0],Ar_Pos[1],Ar_Pos[2],file=MD)
        Pars.Ar_Z.append(Ar_Pos[2])


################################################################################
def Exit(Pt_Pos,Ar_Pos,Pars,timestep):
    if(Ar_Pos[2]>Pars.d or timestep>=Pars.Tt):
        Dump(Pt_Pos,Ar_Pos,Pars,timestep)
        Pars.state=False
    else:
        Dump(Pt_Pos,Ar_Pos,Pars,timestep,Pars.dumpstep)


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
        if(Ar_Pos[axis]<Pars.Box[axis,0]):
            Ar_Pos[axis]+=Pars.Box[axis,2]
        elif(Ar_Pos[axis]>=Pars.Box[axis,1]):
            Ar_Pos[axis]-=Pars.Box[axis,2]
    #Z方向下边界更新
    Pars.Box[2,0]=min(Pt_Pos[:,2])
    Pars.Box[2,2]=Pars.Box[2,1]-Pars.Box[2,0]


    return(Ar_Pos,Pt_Pos)
################################################################################
def Acceleration_period(At_Pos,Pt_Pos,Pars):
    All_Pos=Pt_Pos
    All_Pos=append(All_Pos,[Ar_Pos],axis=0)
    All_type=Pars.Pt_type
    All_type=append(All_type,[Pars.Ar_type])
    All_N=Pars.Pt_N+Pars.Ar_N
    Pt_F=empty(shape=[0,3])
    for i in range(All_N):
        Atom_F=zeros(3)
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
                if(abs(Pair[axis])>Pars.Box[axis,2]-Pars.cutoff):
                    Pair[axis]=Pair[axis]-Pars.Box[axis,2]*Pair[axis]/abs(Pair[axis])
            #周期距离
            Dispair=sqrt(Pair[0]**2+Pair[1]**2+Pair[2]**2)
            if(0<Dispair<=Pars.cutoff):
                Fpair=48*Epair*(Spair**12/Dispair**13-0.5*Spair**6/Dispair**7)
                Atom_F+=Pair*Fpair/Dispair
        if(All_type[i]==1):
            Pt_F=append(Pt_F,[Atom_F],axis=0)
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
#***********************************Others*************************************#
################################################################################

################################################################################
def V_plot(Pars,Pt_Vel):
    nacc=25
    mp_V2=[]
    for i in range(Pars.Pt_N):
        mp_V2.append(Pt_Vel[i,0]**2+Pt_Vel[i,1]**2+Pt_Vel[i,2]**2)
    mp_V=[sqrt(i) for i in mp_V2]
    #mp_V=[[ for i in line] for line in Pt_Vel]+[i**2 for i in Pt_Vel[:,1]]+[i**2 for i in Pt_Vel[:,2]]
    print('x方向平均速度'+str(sum(Pt_Vel[:,0])/Pars.Pt_N))
    print('y方向平均速度'+str(sum(Pt_Vel[:,1])/Pars.Pt_N))
    print('z方向平均速度'+str(sum(Pt_Vel[:,2])/Pars.Pt_N))
    print('温度'+str(sum(mp_V2)*Pars.nd_Velocity**2*Pars.Mass[1]*Pars.nd_Mass/(3*Pars.Pt_N*Pars.kB)))
    test=[int(round(i*nacc)) for i in mp_V]
    #test=[int(round(i*nacc)) for i in Pt_Vel[:,0]]
    #test=[int(round(i*nacc)) for i in Pt_Vel[:,1]]
    #test=[int(round(i*nacc)) for i in Pt_Vel[:,2]]
    maxu=max(test)
    minu=min(test)
    print(maxu/nacc,minu/nacc)
    fu=zeros((maxu-minu+1,2))
    thex=[x/nacc/10 for x in range(minu*10,(maxu+1)*10)]
    #thef=[1/(sqrt(2*Pars.pi)*Pars.mp_V[1]/sqrt(3))*exp(-thexi**2/(2*Pars.mp_V[1]**2/3)) for thexi in thex]
    thefmax=[4*Pars.pi*(3/(2*Pars.pi*Pars.mp_V[1]**2))**(3/2)*exp(-3*thexi**2/(2*Pars.mp_V[1]**2))*thexi**2 for thexi in thex]
    du=minu
    while minu<=du<=maxu:
        fu[du-minu,0]=du/nacc
        fu[du-minu,1]=test.count(du)*nacc/Pars.Pt_N
        du+=1
    print(sum(fu[:,1])/nacc)
    plot(fu[:,0],fu[:,1],'ro',markersize=1,label='Random')
    mpx=[Pars.mp_V[1] for x in range(250)]
    mpy=[y/100 for y in range(250)]
    plot(mpx,mpy,'g--',markersize=1,label='RMS')
    #plot(thex,thef,'b',markersize=1,label='Guass')
    plot(thex,thefmax,'b',markersize=1,label='Maxwell')
    #axis([-3,3,0,0.9])
    legend(loc='upper left')
    xlabel('$v^{*}$')
    ylabel('$f(v^{*})$')
    savefig('Pt_T_Distribution.png',dpi=600)
    show()
