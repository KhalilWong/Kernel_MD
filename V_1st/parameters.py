from numpy import *

class Parameters():

    def __init__(self):
        #物理参数
        self.pi=3.14159265
        self.Mass=[i/6.02*1E-26 for i in [39.95,195.08]]#单位kg
        self.LJ_E=[1.654E-21,5.207E-20,1.093E-21]#单位J
        self.LJ_S=[i*1E-10 for i in [3.40,2.47,2.94]]#单位m
        self.LJ_EF=1
        self.LJ_E.append(self.LJ_E[0]*self.LJ_EF)
        self.fcc_lattice=3.93E-10
        self.kB=1.38E-23
        self.T=[300,300]
        self.mp_V=[sqrt(2*self.kB*self.T[0]/self.Mass[0]),sqrt(3*self.kB*self.T[1]/self.Mass[1])]#气体最概然速率，固体方均根速率
        #无量纲参数
        self.nd_Mass=self.Mass[1]
        self.nd_Energy=self.LJ_E[1]
        self.nd_Length=self.LJ_S[1]
        self.nd_Velocity=sqrt(self.nd_Energy/self.nd_Mass)
        self.nd_Time=self.nd_Length/self.nd_Velocity
        self.nd_Acceleration=self.nd_Energy/(self.nd_Mass*self.nd_Length)
        #无量纲化
        self.Mass=[i/self.nd_Mass for i in self.Mass]
        self.LJ_E=[i/self.nd_Energy for i in self.LJ_E]
        self.LJ_S=[i/self.nd_Length for i in self.LJ_S]
        self.cutoff=10*1E-10/self.nd_Length
        self.fcc_lattice/=self.nd_Length
        self.mp_V=[i/self.nd_Velocity for i in self.mp_V]
        self.d=5.0
        self.spr_k=5000
        self.dt=0.001
        self.Tt=3500000
        #盒子参数
        self.Box=zeros([3,3])
        self.Pt_ePos=[]
        self.Pt_N=0
        self.Pt_type=[]
        self.Ar_N=0
        self.Ar_type=0
        #状态参数
        self.state=True
        self.dumpstep=100
        #结果参数
        self.Ar_Z=[]

        print('*******Parameters Setted!*******')
