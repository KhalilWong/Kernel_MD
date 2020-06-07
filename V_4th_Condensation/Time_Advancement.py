'''
Name:              Time_Advancement.py
Description:       Functions for Advancement of Time in Molecular Dynamics with CUDA
Auther:            KhalilWong
License:           MIT License
Copyright:         Copyright (C) 2019, by KhalilWong
Version:           4.0
Date:              2019/09/11
UpdateLog:         # 2019/09/11
                   > - 根据相互作用强度，计算凝结判据
                   > - 初始化每个分子有效时间，时间推进至凝结判据
Namespace:         https://github.com/KhalilWong
DownloadURL:       https://github.com/KhalilWong/...
UpdateURL:         https://github.com/KhalilWong/...
'''
################################################################################
from numba import cuda, int32, float32
import numba as nb
import numpy as np
import math
import time
import os
import Lib_Ar as LA
################################################################################
@cuda.jit
def Time_Advancement(Type,Elasticity,Pos,Vel,Acc,Box,ZLow,LJ_E,LJ_S,Mass,Pt_argVel,Pt_V2,Pt_T,T,Pt_N,dt,cutoff,spr_k,nd_Velocity,nd_Mass,kB,N):
    tx=cuda.threadIdx.x
    bx=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    pos=tx+tpb*bx
    if pos<N:
        #Verlet_Pos
        for i in range(3):
            Pos[pos,i]+=Vel[pos,i]*dt+0.5*Acc[pos,i]*dt**2
        #Boundary_XY
        for i in range(2):
            if Pos[pos,i]<Box[i,0]:
                Pos[pos,i]+=Box[i,1]-Box[i,0]
                if Elasticity[pos]:
                    Pos[pos,i+3]+=Box[i,1]-Box[i,0]
            elif Pos[pos,i]>=Box[i,1]:
                Pos[pos,i]-=Box[i,1]-Box[i,0]
                if Elasticity[pos]:
                    Pos[pos,i+3]-=Box[i,1]-Box[i,0]
        cuda.atomic.min(ZLow,0,Pos[pos,2])
        cuda.syncthreads()
        Box[2,0]=ZLow[0]
        #Last_Acceleration
        for i in range(3):
            Acc[pos,i+3] = Acc[pos,i]
        #AccelerationCal
        Atom_Fx=0.0
        Atom_Fy=0.0
        Atom_Fz=0.0
        for i in range(N):
            if(Type[pos]==2 and Type[i]==2):
                LJ_pair=1
            elif(Type[pos]==1 and Type[i]==1):
                LJ_pair=0
            else:
                LJ_pair=2
            Epair=LJ_E[LJ_pair]
            Spair=LJ_S[LJ_pair]
            #周期相对位置
            Pairx=Pos[pos,0]-Pos[i,0]
            Pairy=Pos[pos,1]-Pos[i,1]
            Pairz=Pos[pos,2]-Pos[i,2]
            if(abs(Pairx)>=Box[0,1]-Box[0,0]-cutoff):
                Pairx=Pairx-(Box[0,1]-Box[0,0])*Pairx/abs(Pairx)
            if(abs(Pairy)>=Box[1,1]-Box[1,0]-cutoff):
                Pairy=Pairy-(Box[1,1]-Box[1,0])*Pairy/abs(Pairy)
            #周期距离
            Dispair=math.sqrt(Pairx**2+Pairy**2+Pairz**2)
            if(Dispair>0 and Dispair<=cutoff):
                Fpair=48*Epair*(Spair**12/Dispair**13-0.5*Spair**6/Dispair**7)
                Atom_Fx+=Pairx*Fpair/Dispair
                Atom_Fy+=Pairy*Fpair/Dispair
                Atom_Fz+=Pairz*Fpair/Dispair
        if(Elasticity[pos]):
            #弹性恢复力
            Spring_Disx=Pos[pos,0]-Pos[pos,3]
            Spring_Fx=-spr_k*Spring_Disx
            Atom_Fx+=Spring_Fx
            Spring_Disy=Pos[pos,1]-Pos[pos,4]
            Spring_Fy=-spr_k*Spring_Disy
            Atom_Fy+=Spring_Fy
            Spring_Disz=Pos[pos,2]-Pos[pos,5]
            Spring_Fz=-spr_k*Spring_Disz
            Atom_Fz+=Spring_Fz
        Acc[pos,0]=Atom_Fx/Mass[int(Type[pos]-1)]
        Acc[pos,1]=Atom_Fy/Mass[int(Type[pos]-1)]
        Acc[pos,2]=Atom_Fz/Mass[int(Type[pos]-1)]
        #Verlet_Vel
        for i in range(3):
            Vel[pos,i]+=0.5*(Acc[pos,i]+Acc[pos,i+3])*dt
        #Rescale_T
        if Type[pos]==2:
            cuda.atomic.add(Pt_argVel,0,(Vel[pos,0])/Pt_N)
            cuda.atomic.add(Pt_argVel,1,(Vel[pos,1])/Pt_N)
            cuda.atomic.add(Pt_argVel,2,(Vel[pos,2])/Pt_N)
        cuda.syncthreads()
        #只需要热运动速度
        if Type[pos]==2:
            Vel[pos,0]-=Pt_argVel[0]
            Vel[pos,1]-=Pt_argVel[1]
            Vel[pos,2]-=Pt_argVel[2]
            cuda.atomic.add(Pt_V2,0,Vel[pos,0]**2+Vel[pos,1]**2+Vel[pos,2]**2)
        cuda.syncthreads()
        if Type[pos]==2:
            Pt_T[0]=Pt_V2[0]*nd_Velocity**2*Mass[int(Type[pos]-1)]*nd_Mass/(3*Pt_N*kB)
            Vel[pos,0]*=math.sqrt(T[int(Type[pos]-1)]/Pt_T[0])
            Vel[pos,1]*=math.sqrt(T[int(Type[pos]-1)]/Pt_T[0])
            Vel[pos,2]*=math.sqrt(T[int(Type[pos]-1)]/Pt_T[0])
################################################################################
def ReadData(FileName,mode='MD'):
    #
    if(mode=='Lib'):
        with open(FileName,'r') as data:
            InData=data.readlines()
            N=int(InData[1])
            ID=np.zeros(N)
            Type=np.zeros(N)
            Elasticity=np.zeros(N)
            X=np.zeros(N)
            Y=np.zeros(N)
            Z=np.zeros(N)
            VX=np.zeros(N)
            VY=np.zeros(N)
            VZ=np.zeros(N)
            AX=np.zeros(N)
            AY=np.zeros(N)
            AZ=np.zeros(N)
            count=0
            AtomsData=InData[3:]
            for pdata in AtomsData:
                (atomid,atomtype,atomela,x,y,z,vx,vy,vz,ax,ay,az)=pdata.split(' ',11)
                ID[count]=int(atomid)
                Type[count]=int(atomtype)
                Elasticity[count]=int(atomela)
                X[count]=float(x)
                Y[count]=float(y)
                Z[count]=float(z)
                VX[count]=float(vx)
                VY[count]=float(vy)
                VZ[count]=float(vz)
                AX[count]=float(ax)
                AY[count]=float(ay)
                AZ[count]=float(az)
                count+=1
        return(N,ID,Type,Elasticity,X,Y,Z,VX,VY,VZ,AX,AY,AZ)
    elif(mode=='MD'):
        with open(FileName,'r') as data:
            InData=data.readlines()
            TimeStep=int(InData[1])
            N=int(InData[3])
            Box=np.zeros((3,2))
            ID=np.zeros(N)
            Type=np.zeros(N)
            Elasticity=np.zeros(N)
            X=np.zeros(N)
            Y=np.zeros(N)
            Z=np.zeros(N)
            VX=np.zeros(N)
            VY=np.zeros(N)
            VZ=np.zeros(N)
            AX=np.zeros(N)
            AY=np.zeros(N)
            AZ=np.zeros(N)
            XLH=InData[5]
            (BoxXLow,BoxXHigh)=XLH.split()
            Box[0,0]=float(BoxXLow)
            Box[0,1]=float(BoxXHigh)
            YLH=InData[6]
            (BoxYLow,BoxYHigh)=YLH.split()
            Box[1,0]=float(BoxYLow)
            Box[1,1]=float(BoxYHigh)
            ZLH=InData[7]
            (BoxZLow,BoxZHigh)=ZLH.split()
            Box[2,0]=float(BoxZLow)
            Box[2,1]=float(BoxZHigh)
            count=0
            AtomsData=InData[9:]
            for pdata in AtomsData:
                (atomid,atomtype,atomela,x,y,z,vx,vy,vz,ax,ay,az)=pdata.split(' ',11)
                ID[count]=int(atomid)
                Type[count]=int(atomtype)
                Elasticity[count]=int(atomela)
                X[count]=float(x)
                Y[count]=float(y)
                Z[count]=float(z)
                VX[count]=float(vx)
                VY[count]=float(vy)
                VZ[count]=float(vz)
                AX[count]=float(ax)
                AY[count]=float(ay)
                AZ[count]=float(az)
                count+=1
        return(N,Box,ID,Type,Elasticity,X,Y,Z,VX,VY,VZ,AX,AY,AZ)
    elif(mode=='Bal'):
        with open(FileName,'r') as data:
            InData=data.readlines()
            N=int(InData[1])
            ID=np.zeros(N)
            BX=np.zeros(N)
            BY=np.zeros(N)
            BZ=np.zeros(N)
            count=0
            AtomsData=InData[3:]
            for pdata in AtomsData:
                (atomid,Bx,By,Bz)=pdata.split(' ',3)
                ID[count]=int(atomid)
                BX[count]=float(Bx)
                BY[count]=float(By)
                BZ[count]=float(Bz)
                count+=1
        return(N,ID,BX,BY,BZ)
################################################################################
@cuda.jit
def Verlet_Pos(Sid,Pos,Vel,Acc,dt):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        for i in range(3):
            Pos[tid+bid*tpb,i]+=Vel[tid+bid*tpb,i]*dt+0.5*Acc[tid+bid*tpb,i]*dt**2
################################################################################
@cuda.jit
def Boundary_XY(Sid,Pos,Box,Pt_N):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        for i in range(2):
            if Pos[tid+bid*tpb,i]<Box[i,0]:
                Pos[tid+bid*tpb,i]+=Box[i,1]-Box[i,0]
                if tid<Pt_N:
                    Pos[tid+bid*tpb,i+3]+=Box[i,1]-Box[i,0]
            elif Pos[tid+bid*tpb,i]>=Box[i,1]:
                Pos[tid+bid*tpb,i]-=Box[i,1]-Box[i,0]
                if tid<Pt_N:
                    Pos[tid+bid*tpb,i+3]-=Box[i,1]-Box[i,0]
################################################################################
@cuda.jit
def Last_Acceleration(Sid,Acc):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        for i in range(3):
            Acc[tid+bid*tpb,i+3] = Acc[tid+bid*tpb,i]
################################################################################
@cuda.jit
def AccelerationCal(Sid,Acc,Pos,Box,cutoff,LJ_E,LJ_S,spr_k,Mass,Pt_N):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        Acc[tid+bid*tpb,0]=0.0
        Acc[tid+bid*tpb,1]=0.0
        Acc[tid+bid*tpb,2]=0.0
        for i in range(tpb):
            #周期相对位置
            Pairx=Pos[tid+bid*tpb,0]-Pos[i+bid*tpb,0]
            Pairy=Pos[tid+bid*tpb,1]-Pos[i+bid*tpb,1]
            Pairz=Pos[tid+bid*tpb,2]-Pos[i+bid*tpb,2]
            if(abs(Pairx)>=Box[0,1]-Box[0,0]-cutoff):
                Pairx=Pairx-(Box[0,1]-Box[0,0])*Pairx/abs(Pairx)
            if(abs(Pairy)>=Box[1,1]-Box[1,0]-cutoff):
                Pairy=Pairy-(Box[1,1]-Box[1,0])*Pairy/abs(Pairy)
            #
            if(tid<Pt_N and i<Pt_N):
                LJ_pair=1
            elif(tid==Pt_N and i==Pt_N):
                LJ_pair=0
            else:
                LJ_pair=2
            #周期距离
            Dispair=math.sqrt(Pairx**2+Pairy**2+Pairz**2)
            if(Dispair>0 and Dispair<=cutoff):
                Fpair=48*LJ_E[LJ_pair]*(LJ_S[LJ_pair]**12/Dispair**13-0.5*LJ_S[LJ_pair]**6/Dispair**7)
                Acc[tid+bid*tpb,0]+=Pairx*Fpair/Dispair
                Acc[tid+bid*tpb,1]+=Pairy*Fpair/Dispair
                Acc[tid+bid*tpb,2]+=Pairz*Fpair/Dispair
        if(tid<Pt_N):
            #弹性恢复力
            Acc[tid+bid*tpb,0]+=-spr_k*(Pos[tid+bid*tpb,0]-Pos[tid+bid*tpb,3])
            Acc[tid+bid*tpb,1]+=-spr_k*(Pos[tid+bid*tpb,1]-Pos[tid+bid*tpb,4])
            Acc[tid+bid*tpb,2]+=-spr_k*(Pos[tid+bid*tpb,2]-Pos[tid+bid*tpb,5])
            Acc[tid+bid*tpb,0]/=Mass[1]
            Acc[tid+bid*tpb,1]/=Mass[1]
            Acc[tid+bid*tpb,2]/=Mass[1]
        else:
            Acc[tid+bid*tpb,0]/=Mass[0]
            Acc[tid+bid*tpb,1]/=Mass[0]
            Acc[tid+bid*tpb,2]/=Mass[0]
################################################################################
@cuda.jit
def Verlet_Vel(Sid,Vel,Acc,dt):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        for i in range(3):
            if(i==2 and tid==tpb-1):
                OldVel=Vel[tid+bid*tpb,i]
                Vel[tid+bid*tpb,i]+=0.5*(Acc[tid+bid*tpb,i]+Acc[tid+bid*tpb,i+3])*dt
                if(OldVel*Vel[tid+bid*tpb,i]<=0.0 and (OldVel<0.0 or Vel[tid+bid*tpb,i]>0.0)):
                    Sid[bid,3]+=1
            else:
                Vel[tid+bid*tpb,i]+=0.5*(Acc[tid+bid*tpb,i]+Acc[tid+bid*tpb,i+3])*dt
################################################################################
@cuda.jit
def Rescale_T1(Sid,RescaleT_Pars,Vel,Pt_N):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        cuda.atomic.add(RescaleT_Pars,(bid,0),Vel[tid+bid*(tpb+1),0]/Pt_N)
        cuda.atomic.add(RescaleT_Pars,(bid,1),Vel[tid+bid*(tpb+1),1]/Pt_N)
        cuda.atomic.add(RescaleT_Pars,(bid,2),Vel[tid+bid*(tpb+1),2]/Pt_N)
################################################################################
@cuda.jit
def Rescale_T2(Sid,Vel,RescaleT_Pars,nd_Velocity,Mass,nd_Mass,kB,Pt_N):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        Vel[tid+bid*(tpb+1),0]-=RescaleT_Pars[bid,0]
        Vel[tid+bid*(tpb+1),1]-=RescaleT_Pars[bid,1]
        Vel[tid+bid*(tpb+1),2]-=RescaleT_Pars[bid,2]
        cuda.atomic.add(RescaleT_Pars,(bid,3),(Vel[tid+bid*(tpb+1),0]**2+Vel[tid+bid*(tpb+1),1]**2+Vel[tid+bid*(tpb+1),2]**2)*nd_Velocity**2*Mass[1]*nd_Mass/(3*Pt_N*kB))
################################################################################
@cuda.jit
def Rescale_T3(Sid,Vel,T,RescaleT_Pars):
    #
    tid=cuda.threadIdx.x
    bid=cuda.blockIdx.x
    tpb=cuda.blockDim.x
    #
    if(Sid[bid,2]==0):
        Vel[tid+bid*(tpb+1),0]*=math.sqrt(T[1]/RescaleT_Pars[bid,3])
        Vel[tid+bid*(tpb+1),1]*=math.sqrt(T[1]/RescaleT_Pars[bid,3])
        Vel[tid+bid*(tpb+1),2]*=math.sqrt(T[1]/RescaleT_Pars[bid,3])
################################################################################
@cuda.jit
def UpdateRescaleT_Pars(Sid,RescaleT_Pars,Pos,d,Tt,CompleteS,Pt_N,Ar_N):
    #
    bid=cuda.blockIdx.x
    #
    if(Sid[bid,2]==0):
        Sid[bid,1]+=1
        Sid[bid,4]+=1
        RescaleT_Pars[bid,0]=0.0
        RescaleT_Pars[bid,1]=0.0
        RescaleT_Pars[bid,2]=0.0
        RescaleT_Pars[bid,3]=0.0
        if(Pos[(bid+1)*(Pt_N+Ar_N)-1,2]>d):
            Sid[bid,2]=1
            cuda.atomic.add(CompleteS,0,1)
        #elif(Sid[bid,1]>=Tt):
        elif(Sid[bid,4]>=Tt):
            Sid[bid,2]=2
            cuda.atomic.add(CompleteS,0,1)
################################################################################
#@nb.jit(nopython=True,nogil=True)
def SaveOldGetNew(i,j,Lib_N,Block_N,Sid,Pos,Vel,Acc,Ar_X,Ar_Y,Ar_Z,Ar_VX,Ar_VY,Ar_VZ,Ar_AX,Ar_AY,Ar_AZ,Pt_X,Pt_Y,Pt_Z,Pt_BX,Pt_BY,Pt_BZ,Pt_VX,Pt_VY,Pt_VZ,Pt_AX,Pt_AY,Pt_AZ,Pt_N,Ar_N,Zh,dt):
    #
    while(j<Block_N):
        if(Sid[j,2]>0):
            #
            Final_Ar_X=Pos[(j+1)*(Pt_N+Ar_N)-1,0]
            Final_Ar_Y=Pos[(j+1)*(Pt_N+Ar_N)-1,1]
            Final_Ar_Z=Pos[(j+1)*(Pt_N+Ar_N)-1,2]
            Final_Ar_VX=Vel[(j+1)*(Pt_N+Ar_N)-1,0]
            Final_Ar_VY=Vel[(j+1)*(Pt_N+Ar_N)-1,1]
            Final_Ar_VZ=Vel[(j+1)*(Pt_N+Ar_N)-1,2]
            #
            with open('Incident_Reflection.data','a') as IR:
                if(Sid[j,2]==2):
                    Adsorbed='True'
                else:
                    Adsorbed='False'
                print('%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s'%(int(Sid[j,0]),int(Sid[j,1]),int(Sid[j,3]),int(Sid[j,4]),Ar_X[int(Sid[j,0])],Ar_Y[int(Sid[j,0])],Ar_Z[int(Sid[j,0])],Ar_VX[int(Sid[j,0])],Ar_VY[int(Sid[j,0])],Ar_VZ[int(Sid[j,0])],Final_Ar_X,Final_Ar_Y,Final_Ar_Z,Final_Ar_VX,Final_Ar_VY,Final_Ar_VZ,Adsorbed),file=IR)
            #
            if(i<Lib_N):
                Sid[j,0]=i
                Sid[j,1]=0
                Sid[j,2]=0
                Sid[j,3]=0
                Sid[j,4]=0-int((Zh/abs(Ar_VZ[int(Sid[j,0])]))/dt)
                Pos[(j+1)*(Pt_N+Ar_N)-1,0]=Ar_X[int(Sid[j,0])]
                Pos[(j+1)*(Pt_N+Ar_N)-1,1]=Ar_Y[int(Sid[j,0])]
                Pos[(j+1)*(Pt_N+Ar_N)-1,2]=Ar_Z[int(Sid[j,0])]
                Pos[(j+1)*(Pt_N+Ar_N)-1,3]=0.0
                Pos[(j+1)*(Pt_N+Ar_N)-1,4]=0.0
                Pos[(j+1)*(Pt_N+Ar_N)-1,5]=0.0
                Vel[(j+1)*(Pt_N+Ar_N)-1,0]=Ar_VX[int(Sid[j,0])]
                Vel[(j+1)*(Pt_N+Ar_N)-1,1]=Ar_VY[int(Sid[j,0])]
                Vel[(j+1)*(Pt_N+Ar_N)-1,2]=Ar_VZ[int(Sid[j,0])]
                Acc[(j+1)*(Pt_N+Ar_N)-1,0]=Ar_AX[int(Sid[j,0])]
                Acc[(j+1)*(Pt_N+Ar_N)-1,1]=Ar_AY[int(Sid[j,0])]
                Acc[(j+1)*(Pt_N+Ar_N)-1,2]=Ar_AZ[int(Sid[j,0])]
                Acc[(j+1)*(Pt_N+Ar_N)-1,3]=0.0
                Acc[(j+1)*(Pt_N+Ar_N)-1,4]=0.0
                Acc[(j+1)*(Pt_N+Ar_N)-1,5]=0.0
                for k in range(Pt_N):
                    Pos[k+j*(Pt_N+Ar_N),0]=Pt_X[k]
                    Pos[k+j*(Pt_N+Ar_N),1]=Pt_Y[k]
                    Pos[k+j*(Pt_N+Ar_N),2]=Pt_Z[k]
                    Pos[k+j*(Pt_N+Ar_N),3]=Pt_BX[k]
                    Pos[k+j*(Pt_N+Ar_N),4]=Pt_BY[k]
                    Pos[k+j*(Pt_N+Ar_N),5]=Pt_BZ[k]
                    Vel[k+j*(Pt_N+Ar_N),0]=Pt_VX[k]
                    Vel[k+j*(Pt_N+Ar_N),1]=Pt_VY[k]
                    Vel[k+j*(Pt_N+Ar_N),2]=Pt_VZ[k]
                    Acc[k+j*(Pt_N+Ar_N),0]=Pt_AX[k]
                    Acc[k+j*(Pt_N+Ar_N),1]=Pt_AY[k]
                    Acc[k+j*(Pt_N+Ar_N),2]=Pt_AZ[k]
                    Acc[k+j*(Pt_N+Ar_N),3]=0.0
                    Acc[k+j*(Pt_N+Ar_N),4]=0.0
                    Acc[k+j*(Pt_N+Ar_N),5]=0.0
            else:
                Sid[j,2]=-1
            j+=1
            break
        else:
            j+=1
################################################################################
def CheckDump(DumpID1,DumpID2,DumpID3,d_Sid,d_Pos,Box,Block_N,Pt_N,Ar_N):
    Sid=d_Sid.copy_to_host()
    BID1=-1
    BID2=-1
    BID3=-1
    for j in range(Block_N):
        if(Sid[j,0]==DumpID1):
            BID1=j
        elif(Sid[j,0]==DumpID2):
            BID2=j
        elif(Sid[j,0]==DumpID3):
            BID3=j
    if(BID1!=-1 or BID2!=-1 or BID3!=-1):
        Pos=d_Pos.copy_to_host()
    #
    if(BID1!=-1 and Sid[BID1,1]%100==0):
        with open(str(DumpID1)+'_id_Sample.dump','a') as SD:
            print('ITEM: TIMESTEP',file=SD)
            print(Sid[BID1,1],file=SD)
            print('ITEM: NUMBER OF ATOMS',file=SD)
            print(Pt_N+Ar_N,file=SD)
            print('ITEM: BOX BOUNDS pp pp ff',file=SD)
            print(Box[0,0],Box[0,1],file=SD)
            print(Box[1,0],Box[1,1],file=SD)
            print(Box[2,0],Box[2,1],file=SD)
            print('ITEM: ATOMS id type x y z',file=SD)
            for i in range(Pt_N):
                print(i+1,2,Pos[i+BID1*(Pt_N+Ar_N),0],Pos[i+BID1*(Pt_N+Ar_N),1],Pos[i+BID1*(Pt_N+Ar_N),2],file=SD)
            print(Pt_N,1,Pos[(BID1+1)*(Pt_N+Ar_N)-1,0],Pos[(BID1+1)*(Pt_N+Ar_N)-1,1],Pos[(BID1+1)*(Pt_N+Ar_N)-1,2],file=SD)
    if(BID2!=-1 and Sid[BID2,1]%100==0):
        with open(str(DumpID2)+'_id_Sample.dump','a') as SD:
            print('ITEM: TIMESTEP',file=SD)
            print(Sid[BID2,1],file=SD)
            print('ITEM: NUMBER OF ATOMS',file=SD)
            print(Pt_N+Ar_N,file=SD)
            print('ITEM: BOX BOUNDS pp pp ff',file=SD)
            print(Box[0,0],Box[0,1],file=SD)
            print(Box[1,0],Box[1,1],file=SD)
            print(Box[2,0],Box[2,1],file=SD)
            print('ITEM: ATOMS id type x y z',file=SD)
            for i in range(Pt_N):
                print(i+1,2,Pos[i+BID2*(Pt_N+Ar_N),0],Pos[i+BID2*(Pt_N+Ar_N),1],Pos[i+BID2*(Pt_N+Ar_N),2],file=SD)
            print(Pt_N,1,Pos[(BID2+1)*(Pt_N+Ar_N)-1,0],Pos[(BID2+1)*(Pt_N+Ar_N)-1,1],Pos[(BID2+1)*(Pt_N+Ar_N)-1,2],file=SD)
    if(BID3!=-1 and Sid[BID3,1]%100==0):
        with open(str(DumpID3)+'_id_Sample.dump','a') as SD:
            print('ITEM: TIMESTEP',file=SD)
            print(Sid[BID3,1],file=SD)
            print('ITEM: NUMBER OF ATOMS',file=SD)
            print(Pt_N+Ar_N,file=SD)
            print('ITEM: BOX BOUNDS pp pp ff',file=SD)
            print(Box[0,0],Box[0,1],file=SD)
            print(Box[1,0],Box[1,1],file=SD)
            print(Box[2,0],Box[2,1],file=SD)
            print('ITEM: ATOMS id type x y z',file=SD)
            for i in range(Pt_N):
                print(i+1,2,Pos[i+BID3*(Pt_N+Ar_N),0],Pos[i+BID3*(Pt_N+Ar_N),1],Pos[i+BID3*(Pt_N+Ar_N),2],file=SD)
            print(Pt_N,1,Pos[(BID3+1)*(Pt_N+Ar_N)-1,0],Pos[(BID3+1)*(Pt_N+Ar_N)-1,1],Pos[(BID3+1)*(Pt_N+Ar_N)-1,2],file=SD)
################################################################################
def main():
    #
    Dim=LA.Dim_Parameters(195.08/6.02*1E-26,5.207E-20,2.47E-10)
    MD_Pars=LA.MD_Parameters()
    MD_Pars.Non_Dim(Dim)
    Pt_Pars=LA.Atom_Parameters(195.08/6.02*1E-26,5.207E-20,2.47E-10)
    Pt_Pars.Set_T(MD_Pars.kB,300.0)
    Pt_Pars.Non_Dim(Dim)
    Ar_Pars=LA.Atom_Parameters(39.95/6.02*1E-26,1.654E-21,3.40E-10)
    Ar_Pars.Set_T(MD_Pars.kB,300.0,2)
    Ar_Pars.Non_Dim(Dim)
    #
    LJ_E=np.zeros(3)
    LJ_S=np.zeros(3)
    Mass=np.zeros(2)
    T=np.zeros(2)
    LJ_E[0]=Ar_Pars.Epsilon
    LJ_E[1]=Pt_Pars.Epsilon
    LJ_E[2]=math.sqrt(LJ_E[0]*LJ_E[1])
    GS_F=0.1
    LJ_E[2]*=GS_F
    #LJ_E[2]=1.093E-21/Dim.Energy
    LJ_S[0]=Ar_Pars.Sigma
    LJ_S[1]=Pt_Pars.Sigma
    LJ_S[2]=(LJ_S[0]+LJ_S[1])/2
    Mass[0]=Ar_Pars.Mass
    Mass[1]=Pt_Pars.Mass
    T[0]=Ar_Pars.T
    T[1]=Pt_Pars.T
    #
    (Lib_N,Ar_ID,Ar_Type,Ar_Elasticity,Ar_X,Ar_Y,Ar_Z,Ar_VX,Ar_VY,Ar_VZ,Ar_AX,Ar_AY,Ar_AZ)=ReadData('Lib_Ar.data','Lib')
    (Pt_N,Box,Pt_ID,Pt_Type,Pt_Elasticity,Pt_X,Pt_Y,Pt_Z,Pt_VX,Pt_VY,Pt_VZ,Pt_AX,Pt_AY,Pt_AZ)=ReadData('Pt_Wall.data')
    (Pt_BN,Pt_BID,Pt_BX,Pt_BY,Pt_BZ)=ReadData('Pt_Wall.balance','Bal')
    Ar_N=1
    #
    Block_N=1000
    Pos=np.zeros((Block_N*(Pt_N+Ar_N),6))
    Vel=np.zeros((Block_N*(Pt_N+Ar_N),3))
    Acc=np.zeros((Block_N*(Pt_N+Ar_N),6))
    Sid=np.zeros((Block_N,5))                                                   #全局ID；已运行时间步；状态（-1：无新的算例，线程关闭；0：未完成；1：散射；2：吸附）；与壁面碰撞次数；有效相互作用时间步
    RescaleT_Pars=np.zeros((Block_N,4))
    for i in range(Block_N):
        Pos[(i+1)*(Pt_N+Ar_N)-1,0]=Ar_X[i]
        Pos[(i+1)*(Pt_N+Ar_N)-1,1]=Ar_Y[i]
        Pos[(i+1)*(Pt_N+Ar_N)-1,2]=Ar_Z[i]
        Pos[(i+1)*(Pt_N+Ar_N)-1,3]=0.0
        Pos[(i+1)*(Pt_N+Ar_N)-1,4]=0.0
        Pos[(i+1)*(Pt_N+Ar_N)-1,5]=0.0
        Vel[(i+1)*(Pt_N+Ar_N)-1,0]=Ar_VX[i]
        Vel[(i+1)*(Pt_N+Ar_N)-1,1]=Ar_VY[i]
        Vel[(i+1)*(Pt_N+Ar_N)-1,2]=Ar_VZ[i]
        Acc[(i+1)*(Pt_N+Ar_N)-1,0]=Ar_AX[i]
        Acc[(i+1)*(Pt_N+Ar_N)-1,1]=Ar_AY[i]
        Acc[(i+1)*(Pt_N+Ar_N)-1,2]=Ar_AZ[i]
        Acc[(i+1)*(Pt_N+Ar_N)-1,3]=0.0
        Acc[(i+1)*(Pt_N+Ar_N)-1,4]=0.0
        Acc[(i+1)*(Pt_N+Ar_N)-1,5]=0.0
        Sid[i,0]=i
        Sid[i,1]=0                                                              #未减去以入射速度作为平均速度的无效时间步长
        Sid[i,2]=0
        Sid[i,3]=0
        Sid[i,4]=0-int((MD_Pars.BoxZHigh/abs(Ar_VZ[i]))/MD_Pars.dt)             #减去以入射速度作为平均速度的无效时间步长
        for j in range(Pt_N):
            Pos[j+i*(Pt_N+Ar_N),0]=Pt_X[j]
            Pos[j+i*(Pt_N+Ar_N),1]=Pt_Y[j]
            Pos[j+i*(Pt_N+Ar_N),2]=Pt_Z[j]
            Pos[j+i*(Pt_N+Ar_N),3]=Pt_BX[j]
            Pos[j+i*(Pt_N+Ar_N),4]=Pt_BY[j]
            Pos[j+i*(Pt_N+Ar_N),5]=Pt_BZ[j]
            Vel[j+i*(Pt_N+Ar_N),0]=Pt_VX[j]
            Vel[j+i*(Pt_N+Ar_N),1]=Pt_VY[j]
            Vel[j+i*(Pt_N+Ar_N),2]=Pt_VZ[j]
            Acc[j+i*(Pt_N+Ar_N),0]=Pt_AX[j]
            Acc[j+i*(Pt_N+Ar_N),1]=Pt_AY[j]
            Acc[j+i*(Pt_N+Ar_N),2]=Pt_AZ[j]
            Acc[j+i*(Pt_N+Ar_N),3]=0.0
            Acc[j+i*(Pt_N+Ar_N),4]=0.0
            Acc[j+i*(Pt_N+Ar_N),5]=0.0
    CompleteS=np.zeros(1)
    OldCompleteS=np.zeros(1)
    CompleteS[0]=Block_N-1
    #
    d_Pos=cuda.to_device(Pos)
    d_Vel=cuda.to_device(Vel)
    d_Acc=cuda.to_device(Acc)
    d_Sid=cuda.to_device(Sid)
    d_RescaleT_Pars=cuda.to_device(RescaleT_Pars)
    #
#    Samples_Parallel(d_All_N,d_Data,d_AllData,d_Box,d_LJ_E,d_LJ_S,d_Mass,d_T,d_All_Pars)
#    kern=Samples_Parallel.specialize(d_All_N,d_Data,d_AllData,d_Box,d_LJ_E,d_LJ_S,d_Mass,d_T,d_All_Pars)
#    with open('info.ptx','w') as out:
#        print(kern._func.get_info(),file=out)
#    with open('inspect_asm.ptx','w') as out:
#        print(str(kern.inspect_asm()),file=out)
#    with open('inspect_types.ptx','w') as out:
#        kern.inspect_types(file=out)
#    with open('inspect_llvm.ptx','w') as out:
#        print(str(kern.inspect_llvm()),file=out)
    #
    State=True
    with open('Incident_Reflection.data','w') as IR:
        print('ID\tTt\tCN\tTtmVt\tIX\tIY\tIZ\tIVX\tIVY\tIVZ\tFX\tFY\tFZ\tFVX\tFVY\tFVZ\tState',file=IR)
    TotalLoop=0
    #MD_Pars.Period=500000
    Art_Cri_t=27
    MD_Pars.Period=int((Art_Cri_t/np.sqrt(GS_F))/MD_Pars.dt)
    #
    #DumpID1=np.random.randint(Lib_N)
    #DumpID2=np.random.randint(Lib_N)
    #while(DumpID2==DumpID1):
    #    DumpID2=np.random.randint(Lib_N)
    #DumpID3=np.random.randint(Lib_N)
    #while(DumpID3==DumpID1 or DumpID3==DumpID2):
    #    DumpID3=np.random.randint(Lib_N)
    #CheckDump(DumpID1,DumpID2,DumpID3,d_Sid,d_Pos,Box,Block_N,Pt_N,Ar_N)
    #
    start_time=time.time()
    #
    while(State):
        Verlet_Pos[Block_N,Pt_N+Ar_N](d_Sid,d_Pos,d_Vel,d_Acc,MD_Pars.dt)
        Boundary_XY[Block_N,Pt_N+Ar_N](d_Sid,d_Pos,Box,Pt_N)
        Last_Acceleration[Block_N,Pt_N+Ar_N](d_Sid,d_Acc)
        AccelerationCal[Block_N,Pt_N+Ar_N](d_Sid,d_Acc,d_Pos,Box,MD_Pars.cutoff,LJ_E,LJ_S,MD_Pars.Spr_K,Mass,Pt_N)
        Verlet_Vel[Block_N,Pt_N+Ar_N](d_Sid,d_Vel,d_Acc,MD_Pars.dt)
        #Rescale_T1[Block_N,Pt_N](d_Sid,d_RescaleT_Pars,d_Vel,Pt_N)
        #Rescale_T2[Block_N,Pt_N](d_Sid,d_Vel,d_RescaleT_Pars,Dim.Velocity,Mass,Dim.Mass,MD_Pars.kB,Pt_N)
        #Rescale_T3[Block_N,Pt_N](d_Sid,d_Vel,T,d_RescaleT_Pars)
        OldCompleteS[0]=CompleteS[0]
        UpdateRescaleT_Pars[Block_N,1](d_Sid,d_RescaleT_Pars,d_Pos,MD_Pars.BoxZHigh,MD_Pars.Period,CompleteS,Pt_N,Ar_N)
        #
        #CheckDump(DumpID1,DumpID2,DumpID3,d_Sid,d_Pos,Box,Block_N,Pt_N,Ar_N)
        #
        if(CompleteS[0]!=OldCompleteS[0]):
            j=0
            for i in range(int(OldCompleteS[0])+1,int(CompleteS[0])+1):
                #
                if(i==OldCompleteS[0]+1):
                    Sid=d_Sid.copy_to_host()
                    Pos=d_Pos.copy_to_host()
                    Vel=d_Vel.copy_to_host()
                    Acc=d_Acc.copy_to_host()
                #
                SaveOldGetNew(i,j,Lib_N,Block_N,Sid,Pos,Vel,Acc,Ar_X,Ar_Y,Ar_Z,Ar_VX,Ar_VY,Ar_VZ,Ar_AX,Ar_AY,Ar_AZ,Pt_X,Pt_Y,Pt_Z,Pt_BX,Pt_BY,Pt_BZ,Pt_VX,Pt_VY,Pt_VZ,Pt_AX,Pt_AY,Pt_AZ,Pt_N,Ar_N,MD_Pars.BoxZHigh,MD_Pars.dt)
                if(i==CompleteS[0]):
                    if(int(OldCompleteS[0])+1<Lib_N):
                        d_Pos=cuda.to_device(Pos)
                        d_Vel=cuda.to_device(Vel)
                        d_Acc=cuda.to_device(Acc)
                        d_Sid=cuda.to_device(Sid)
                    else:
                        d_Sid=cuda.to_device(Sid)
        #
        State=False
        for i in range(Block_N):
            if(Sid[i,2]==0):
                State=True
                break
        TotalLoop+=1
        step_time=time.time()
        print('Total Loops: %d. Complete Samples: %d. Average Time: %f Seconds Per Loop'%(TotalLoop,int(CompleteS[0])-Block_N+1,(step_time-start_time)/TotalLoop),end='\r')
    #
    end_time=time.time()
    print('Total Loops: %d. Total Time: %dd %dh %dm %ds. Complete Samples: %d. Average Time: %f Seconds Per Loop.'%(TotalLoop,int(int(end_time-start_time)/86400),int(int(end_time-start_time)%86400/3600),int(int(end_time-start_time)%3600/60),int(end_time-start_time)%60,int(CompleteS[0])-Block_N+1,(end_time-start_time)/TotalLoop))
################################################################################
if __name__ == '__main__':
    #
    main()
    os.system("pause")
