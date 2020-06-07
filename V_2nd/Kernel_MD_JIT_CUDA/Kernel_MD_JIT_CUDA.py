import time
import numpy as np
import math
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_uniform_float64,xoroshiro128p_normal_float64
import matplotlib.pyplot as mpl

################################################################################
spec=[
    ('pi',nb.float64),
    ('Mass',nb.float64[:]),
    ('LJ_E',nb.float64[:]),
    ('LJ_S',nb.float64[:]),
    ('fcc_lattice',nb.float64),
    ('kB',nb.float64),
    ('T',nb.float64[:]),
    ('mp_V',nb.float64[:]),
    ('nd_Mass',nb.float64),
    ('nd_Energy',nb.float64),
    ('nd_Length',nb.float64),
    ('nd_Velocity',nb.float64),
    ('nd_Time',nb.float64),
    ('nd_Acceleration',nb.float64),
    ('cutoff',nb.float64),
    ('d',nb.float64),
    ('spr_k',nb.float64),
    ('dt',nb.float64),
    ('Tt',nb.int64),
    ('Pt_I',nb.int64),
    ('Pt_J',nb.int64),
    ('Pt_K',nb.int64),
    ('Pt_N',nb.int64),
    ('Ar_N',nb.int64),
    ('All_N',nb.int64),
    ('Box',nb.float64[:,:]),
    ('Pt_ePos',nb.float64[:,:]),
    ('All_type',nb.float64[:]),
    ('state',nb.boolean),
    ('dumpstep',nb.int64),
    ('Pt_argVel',nb.float64[:]),
    ('Pt_V2',nb.float64[:]),
]

@nb.jitclass(spec)
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
        self.Pt_N=4*self.Pt_I*self.Pt_J*self.Pt_K
        self.Ar_N=1
        self.All_N=self.Pt_N+self.Ar_N
        self.Box=np.zeros((3,3))
        self.Pt_ePos=np.zeros((self.Pt_N,3))
        self.All_type=np.zeros(self.All_N)
        #状态参数
        self.state=True
        self.dumpstep=100


################################################################################
def Initialization(Pars):
    Pars.Box[0,0]=0
    Pars.Box[0,1]=Pars.Pt_I*Pars.fcc_lattice
    Pars.Box[0,2]=Pars.Box[0,1]-Pars.Box[0,0]
    Pars.Box[1,0]=0
    Pars.Box[1,1]=Pars.Pt_J*Pars.fcc_lattice
    Pars.Box[1,2]=Pars.Box[1,1]-Pars.Box[1,0]
    Pars.Box[2,0]=-(Pars.Pt_K-0.5)*Pars.fcc_lattice
    Pars.Box[2,1]=Pars.d
    Pars.Box[2,2]=Pars.Box[2,1]-Pars.Box[2,0]
    print('计算区域X: ',Pars.Box[0,0],', ',Pars.Box[0,1])
    print('计算区域Y: ',Pars.Box[1,0],', ',Pars.Box[1,1])
    print('计算区域Z: ',Pars.Box[2,0],', ',Pars.Box[2,1])
    timestep=0
    #位置，速度初始化
    All_Pos=np.zeros((Pars.All_N,3))
    All_Vel=np.zeros((Pars.All_N,3))
    d_All_Pos=cuda.to_device(All_Pos)
    d_All_Vel=cuda.to_device(All_Vel)
    seedt=str(int(time.time()))
    rng_states=create_xoroshiro128p_states(Pars.All_N,seedt)
    Initialization_Kernel[1,Pars.All_N](d_All_Pos,d_All_Vel,Pars.All_type,Pars.Pt_N,Pars.Pt_I,Pars.Pt_J,Pars.Pt_K,Pars.fcc_lattice,rng_states,Pars.mp_V,Pars.Box,Pars.pi)
    #首次位置周期
    temp=np.array([np.nan])
    Pos_period[1,Pars.All_N](d_All_Pos,Pars.All_N,Pars.Box,Pars.All_type,Pars.Pt_ePos,temp)
    #首次控温
    Pt_argVel=np.zeros(3)
    Pt_V2=np.zeros(1)
    Pt_T=np.zeros(1)
    rescale_T[1,Pars.Pt_N](d_All_Vel,Pt_argVel,Pars.Pt_N,Pt_V2,Pt_T,Pars.nd_Velocity,Pars.Mass,Pars.nd_Mass,Pars.kB,Pars.T)
    #首次加速度周期
    All_Acc=np.zeros((Pars.All_N,3))
    d_All_Acc=cuda.to_device(All_Acc)
    Acceleration_period[1,Pars.All_N](d_All_Pos,d_All_Acc,Pars.All_N,Pars.All_type,Pars.LJ_E,Pars.LJ_S,Pars.Box,Pars.cutoff,Pars.Pt_ePos,Pars.spr_k,Pars.Mass)
    All_Pos=d_All_Pos.copy_to_host()
    All_Vel=d_All_Vel.copy_to_host()
    All_Acc=d_All_Acc.copy_to_host()
    Pars.Pt_ePos=All_Pos[:Pars.Pt_N,:]
    #初始信息
    print('Created ',Pars.Pt_N,' Pt')
    print('Created ',Pars.Ar_N,' Ar')
    print('Pt整体x方向平均速度',Pt_argVel[0])
    print('Pt整体y方向平均速度',Pt_argVel[1])
    print('Pt整体z方向平均速度',Pt_argVel[2])
    print('Pt温度',Pt_T[0])
    print('Ar入射速度:',All_Vel[Pars.Pt_N,0],',',All_Vel[Pars.Pt_N,1],',',All_Vel[Pars.Pt_N,2])
    print('*******Model Initialization Done!*******')

    return(All_Pos,All_Vel,All_Acc,timestep)


################################################################################
@cuda.jit
def Initialization_Kernel(All_Pos,All_Vel,All_type,Pt_N,Pt_I,Pt_J,Pt_K,fcc_lattice,rng_states,mp_V,Box,pi):
    tx=cuda.threadIdx.x
    bx=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    t_pos=tx+tpb*bx
    if(t_pos<Pt_N):#Pt
        i=int(t_pos/(2*Pt_J*Pt_K))
        jk=t_pos%(2*Pt_J*Pt_K)
        j=int(jk/Pt_K)
        k=jk%Pt_K
        if(((i%2)+(j%2))%2==0):
            k=2*k
        else:
            k=2*k+1
        All_Pos[t_pos,0]=i/2*fcc_lattice
        All_Pos[t_pos,1]=j/2*fcc_lattice
        All_Pos[t_pos,2]=(k/2-2.5)*fcc_lattice
        for dim in range(3):
            R=0
            while R==0:
                R=xoroshiro128p_normal_float64(rng_states, t_pos)
            All_Vel[t_pos,dim]=mp_V[1]/math.sqrt(3.0)*R#高斯分布，平均值为0，方差=方均根速度用来控温
        All_type[t_pos]=1.0
    if(t_pos==Pt_N):#Ar
        Rx=xoroshiro128p_uniform_float64(rng_states, t_pos)
        Ry=xoroshiro128p_uniform_float64(rng_states, t_pos)
        All_Pos[t_pos,0]=Box[0,0]+Box[0,2]*Rx
        All_Pos[t_pos,1]=Box[1,0]+Box[1,2]*Ry
        All_Pos[t_pos,2]=Box[2,1]
        R1=0
        while R1==0:
            R1=xoroshiro128p_uniform_float64(rng_states, t_pos)
        R2=0
        while R2==0:
            R2=xoroshiro128p_uniform_float64(rng_states, t_pos)
        All_Vel[t_pos,0]=mp_V[0]*math.sqrt(-math.log(R1))*math.cos(2*pi*R2)#Maxwell分布
        R1=0
        while R1==0:
            R1=xoroshiro128p_uniform_float64(rng_states, t_pos)
        R2=0
        while R2==0:
            R2=xoroshiro128p_uniform_float64(rng_states, t_pos)
        All_Vel[t_pos,1]=mp_V[0]*math.sqrt(-math.log(R1))*math.sin(2*pi*R2)
        R1=0
        while R1==0:
            R1=xoroshiro128p_uniform_float64(rng_states, t_pos)
        R2=0
        while R2==0:
            R2=xoroshiro128p_uniform_float64(rng_states, t_pos)
        All_Vel[t_pos,2]=-mp_V[0]*math.sqrt(-math.log(R1))
        All_type[t_pos]=0.0
    cuda.syncthreads()


################################################################################
@cuda.jit
def Pos_period(All_Pos,All_N,Box,All_type,Pt_ePos,temp):
    tx=cuda.threadIdx.x
    bx=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    t_pos=tx+tpb*bx
    if(t_pos<All_N):
        #X,Y方向周期
        for axis in range(2):
            if(All_Pos[t_pos,axis]<Box[axis,0]):
                All_Pos[t_pos,axis]+=Box[axis,2]
                if(All_type[t_pos]==1):
                    Pt_ePos[t_pos,axis]+=Box[axis,2]
            elif(All_Pos[t_pos,axis]>=Box[axis,1]):
                All_Pos[t_pos,axis]-=Box[axis,2]
                if(All_type[t_pos]==1):
                    Pt_ePos[t_pos,axis]-=Box[axis,2]
        #Z方向下边界更新
        cuda.atomic.min(temp,0,All_Pos[t_pos,2])
        cuda.syncthreads()
        Box[2,0]=temp[0]
        #Pars.Box[2,2]=Pars.Box[2,1]-Pars.Box[2,0]


################################################################################
@cuda.jit
def Acceleration_period(All_Pos,All_Acc,All_N,All_type,LJ_E,LJ_S,Box,cutoff,Pt_ePos,spr_k,Mass):
    tx=cuda.threadIdx.x
    bx=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    t_pos=tx+tpb*bx
    if(t_pos<All_N):
        Atom_Fx=0.
        Atom_Fy=0.
        Atom_Fz=0.
        for j in range(All_N):
            if(All_type[t_pos]==1 and All_type[j]==1):
                LJ_pair=1
            elif(All_type[t_pos]==0 and All_type[j]==0):
                LJ_pair=0
            else:
                LJ_pair=2
            Epair=LJ_E[LJ_pair]
            Spair=LJ_S[LJ_pair]
            #周期相对位置
            Pairx=All_Pos[t_pos,0]-All_Pos[j,0]
            Pairy=All_Pos[t_pos,1]-All_Pos[j,1]
            Pairz=All_Pos[t_pos,2]-All_Pos[j,2]
            if(abs(Pairx)>=Box[0,2]-cutoff):
                Pairx=Pairx-Box[0,2]*Pairx/abs(Pairx)
            if(abs(Pairy)>=Box[1,2]-cutoff):
                Pairy=Pairy-Box[1,2]*Pairy/abs(Pairy)
            #周期距离
            Dispair=math.sqrt(Pairx**2+Pairy**2+Pairz**2)
            if(Dispair>0 and Dispair<=cutoff):
                Fpair=48*Epair*(Spair**12/Dispair**13-0.5*Spair**6/Dispair**7)
                Atom_Fx+=Pairx*Fpair/Dispair
                Atom_Fy+=Pairy*Fpair/Dispair
                Atom_Fz+=Pairz*Fpair/Dispair
        if(All_type[t_pos]==1):
            #Pt弹性恢复力
            Spring_Disx=All_Pos[t_pos,0]-Pt_ePos[t_pos,0]
            Spring_Fx=-spr_k*Spring_Disx
            Pt_Fx=Atom_Fx+Spring_Fx
            All_Acc[t_pos,0]=Pt_Fx/Mass[1]
            Spring_Disy=All_Pos[t_pos,1]-Pt_ePos[t_pos,1]
            Spring_Fy=-spr_k*Spring_Disy
            Pt_Fy=Atom_Fy+Spring_Fy
            All_Acc[t_pos,1]=Pt_Fy/Mass[1]
            Spring_Disz=All_Pos[t_pos,2]-Pt_ePos[t_pos,2]
            Spring_Fz=-spr_k*Spring_Disz
            Pt_Fz=Atom_Fz+Spring_Fz
            All_Acc[t_pos,2]=Pt_Fz/Mass[1]
        else:
            Ar_Fx=Atom_Fx
            All_Acc[t_pos,0]=Ar_Fx/Mass[0]
            Ar_Fy=Atom_Fy
            All_Acc[t_pos,1]=Ar_Fy/Mass[0]
            Ar_Fz=Atom_Fz
            All_Acc[t_pos,2]=Ar_Fz/Mass[0]
        cuda.syncthreads()


################################################################################
@cuda.jit
def Verlet_Pos(All_Pos,All_Vel,All_Acc,All_N,dt):
    tx=cuda.threadIdx.x
    bx=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    t_pos=tx+tpb*bx
    if(t_pos<All_N):
        for axis in range(3):
            All_Pos[t_pos,axis]+=All_Vel[t_pos,axis]*dt+0.5*All_Acc[t_pos,axis]*dt**2
        cuda.syncthreads()


################################################################################
@cuda.jit
def Verlet_Vel(All_Vel,All_Acc_temp,All_Acc,All_N,dt):
    tx=cuda.threadIdx.x
    bx=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    t_pos=tx+tpb*bx
    if(t_pos<All_N):
        for axis in range(3):
            All_Vel[t_pos,axis]+=0.5*(All_Acc_temp[t_pos,axis]+All_Acc[t_pos,axis])*dt
        cuda.syncthreads()


################################################################################
@cuda.jit
def rescale_T(All_Vel,Pt_argVel,Pt_N,Pt_V2,Pt_T,nd_Velocity,Mass,nd_Mass,kB,T):
    tx=cuda.threadIdx.x
    bx=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    t_pos=tx+tpb*bx
    if(t_pos<Pt_N):
        cuda.atomic.add(Pt_argVel,0,(All_Vel[t_pos,0])/Pt_N)
        cuda.atomic.add(Pt_argVel,1,(All_Vel[t_pos,1])/Pt_N)
        cuda.atomic.add(Pt_argVel,2,(All_Vel[t_pos,2])/Pt_N)
        cuda.syncthreads()
        #只需要热运动速度
        All_Vel[t_pos,0]-=Pt_argVel[0]
        All_Vel[t_pos,1]-=Pt_argVel[1]
        All_Vel[t_pos,2]-=Pt_argVel[2]
        cuda.atomic.add(Pt_V2,0,All_Vel[t_pos,0]**2+All_Vel[t_pos,1]**2+All_Vel[t_pos,2]**2)
        cuda.syncthreads()
        Pt_T[0]=Pt_V2[0]*nd_Velocity**2*Mass[1]*nd_Mass/(3*Pt_N*kB)
        All_Vel[t_pos,0]*=math.sqrt(T[1]/Pt_T[0])
        All_Vel[t_pos,1]*=math.sqrt(T[1]/Pt_T[0])
        All_Vel[t_pos,2]*=math.sqrt(T[1]/Pt_T[0])
        cuda.syncthreads()


################################################################################
def Dump(All_Pos,Ar_Z,Ar_t,Pars,timestep,dumpstep=1):
    if(timestep%dumpstep==0):
        with open('Kernel_MD_JIT_CUDA.dump','a') as MD:
            print('ITEM: TIMESTEP',file=MD)
            print(timestep,file=MD)
            print('ITEM: NUMBER OF ATOMS',file=MD)
            print(Pars.All_N,file=MD)
            print('ITEM: BOX BOUNDS pp pp ff',file=MD)
            print(Pars.Box[0,0],Pars.Box[0,1],file=MD)
            print(Pars.Box[1,0],Pars.Box[1,1],file=MD)
            print(Pars.Box[2,0],Pars.Box[2,1],file=MD)
            print('ITEM: ATOMS id type x y z',file=MD)
            for i in range(Pars.All_N):
                print(i+1,int(Pars.All_type[i])+1,All_Pos[i,0],All_Pos[i,1],All_Pos[i,2],file=MD)

        Ar_Z.append(All_Pos[Pars.Pt_N,2])
        Ar_t.append(timestep*Pars.dt)

    return(Ar_Z,Ar_t)


################################################################################
def Exit(All_Pos,Ar_Z,Ar_t,Pars,timestep):
    if(All_Pos[Pars.Pt_N,2]>Pars.d or timestep>=Pars.Tt):
        Pars.state=False
        (Ar_Z,Ar_t)=Dump(All_Pos,Ar_Z,Ar_t,Pars,timestep)
    else:
        (Ar_Z,Ar_t)=Dump(All_Pos,Ar_Z,Ar_t,Pars,timestep,Pars.dumpstep)


################################################################################
#***********************************Others*************************************#
################################################################################

################################################################################
def Fplot(Pars,All_Vel,Ar_Z,Ar_t,timestep,argtime):
    nacc=25
    mp_V2=[]
    for i in range(Pars.Pt_N):
        mp_V2.append(All_Vel[i,0]**2+All_Vel[i,1]**2+All_Vel[i,2]**2)
    mp_V=[math.sqrt(i) for i in mp_V2]
    print('x方向平均速度'+str(np.sum(All_Vel[:Pars.Pt_N,0])/Pars.Pt_N))
    print('y方向平均速度'+str(np.sum(All_Vel[:Pars.Pt_N,1])/Pars.Pt_N))
    print('z方向平均速度'+str(np.sum(All_Vel[:Pars.Pt_N,2])/Pars.Pt_N))
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
(All_Pos,All_Vel,All_Acc,timestep)=Initialization(Pars)
Ar_Z=[]
Ar_t=[]
Dump(All_Pos,Ar_Z,Ar_t,Pars,timestep)
start_time=time.time()
while(Pars.state):
    d_All_Pos=cuda.to_device(All_Pos)
    d_All_Vel=cuda.to_device(All_Vel)
    d_All_Acc=cuda.to_device(All_Acc)
    Verlet_Pos[1,Pars.All_N](d_All_Pos,d_All_Vel,d_All_Acc,Pars.All_N,Pars.dt)
    temp=np.array([np.nan])
    Pos_period[1,Pars.All_N](d_All_Pos,Pars.All_N,Pars.Box,Pars.All_type,Pars.Pt_ePos,temp)
    d_All_Acc_temp=d_All_Acc
    Acceleration_period[1,Pars.All_N](d_All_Pos,d_All_Acc,Pars.All_N,Pars.All_type,Pars.LJ_E,Pars.LJ_S,Pars.Box,Pars.cutoff,Pars.Pt_ePos,Pars.spr_k,Pars.Mass)
    Verlet_Vel[1,Pars.All_N](d_All_Vel,d_All_Acc_temp,d_All_Acc,Pars.All_N,Pars.dt)
    Pt_argVel=np.zeros(3)
    Pt_V2=np.zeros(1)
    Pt_T=np.zeros(1)
    rescale_T[1,Pars.Pt_N](d_All_Vel,Pt_argVel,Pars.Pt_N,Pt_V2,Pt_T,Pars.nd_Velocity,Pars.Mass,Pars.nd_Mass,Pars.kB,Pars.T)
    timestep+=1
    All_Pos=d_All_Pos.copy_to_host()
    All_Vel=d_All_Vel.copy_to_host()
    All_Acc=d_All_Acc.copy_to_host()
    Exit(All_Pos,Ar_Z,Ar_t,Pars,timestep)
    argtime=Bar_show(Pars,timestep,start_time,'Kernel_MD_JIT_CUDA')
Bar_close(start_time,'Kernel_MD_JIT_CUDA')
Fplot(Pars,All_Vel,Ar_Z,Ar_t,timestep,argtime)
