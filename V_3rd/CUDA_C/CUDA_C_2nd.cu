﻿#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include<algorithm>
using namespace std;
const int Lib_N=100000;
const int SampleParallel_N=1000;
const int Atom_type_N=2;
const int Pair_type_N=3;
const int Pt_I = 6, Pt_J = 6, Pt_K = 3;
const int Pt_N=4*Pt_I*Pt_J*Pt_K;
const int Ar_N=1;

class Parameters {
public:
	int Tt, dumpstep;
	double Mass[Atom_type_N], T[Atom_type_N], mp_V[Atom_type_N], LJ_E[Pair_type_N], LJ_S[Pair_type_N], Box[3][2], Pt_ePos[Pt_N][3];
	double PI, kB, fcc_lattice, nd_Mass, nd_Energy, nd_Length, nd_Velocity, nd_Time, nd_Acceleration, cutoff, d, spr_k, dt;
	bool state;
	void Init();
	void Initialization(int All_type[][], double All_Pos[][], double All_Vel[][], double All_Acc[][]);
	void Dump(int All_type[], double All_Pos[][], int timestep, int ds = 1);
	void Exit(int All_type[], double All_Pos[][], int timestep);
};
void Parameters::Init() {
	//物理参数
	PI = 3.14159265;
	kB = 1.38E-23;
	Mass[0] = 39.95 / 6.02 * 1E-26;//单位kg
	Mass[1] = 195.08 / 6.02 * 1E-26;
	LJ_E[0] = 1.654E-21;//单位J
	LJ_E[1] = 5.207E-20;
	LJ_E[2] = 1.093E-21;
	LJ_S[0] = 3.40 * 1E-10;//单位m
	LJ_S[1] = 2.47 * 1E-10;
	LJ_S[2] = 2.94 * 1E-10;
	fcc_lattice = 3.93E-10;
	T[0] = 300.;
	T[1] = 300.;
	mp_V[0] = sqrt(2 * kB*T[0] / Mass[0]);//气体最概然速率
	mp_V[1] = sqrt(3 * kB*T[1] / Mass[1]);//固体方均根速率
	//无量纲参数
	nd_Mass = Mass[1];
	nd_Energy = LJ_E[1];
	nd_Length = LJ_S[1];
	nd_Velocity = sqrt(nd_Energy / nd_Mass);
	nd_Time = nd_Length / nd_Velocity;
	nd_Acceleration = nd_Energy / (nd_Mass * nd_Length);
	//无量纲化
	Mass[0] /= nd_Mass;
	Mass[1] /= nd_Mass;
	LJ_E[0] /= nd_Energy;
	LJ_E[1] /= nd_Energy;
	LJ_E[2] /= nd_Energy;
	LJ_S[0] /= nd_Length;
	LJ_S[1] /= nd_Length;
	LJ_S[2] /= nd_Length;
	cutoff = 10 * 1E-10 / nd_Length;
	fcc_lattice /= nd_Length;
	mp_V[0] /= nd_Velocity;
	mp_V[1] /= nd_Velocity;
	d = 5.0;
	spr_k = 5000.;
	dt = 0.001;
	Rt = 100;
	Tt = 3500000;
	dumpstep = 100;
	//盒子参数
	//状态参数
	state = true;
	Block_N=1;

	cout << "*******Parameters Initialized!*******\n";
}

/******************************************************************************/
void Parameters::Initialization(int All_type[], double All_Pos[][], double All_Vel[][], double All_Acc[][]) {
	double *temp, Pt_argVel[3], *Pt_V2, *Pt_T;
	int *d_All_type;
	double *d_All_Pos,*d_All_Vel,*d_All_Acc;
	__global__ void Initialization_Kernel(int *All_type,double *All_Pos,double *All_Vel,double fcc_lattice,double *mp_V,double *Box,double PI);
	__global__ void Pos_period(double *All_Pos,double *Box,double *Pt_ePos,double *temp);
	__global__ void rescale_T(double *All_Vel,double *Pt_argVel,double *Pt_V2,double *Pt_T,double nd_Velocity,double *Mass,double nd_Mass,double kB,double *T);
	__global__ void Acceleration_period(double *All_Pos,double *All_Acc,double *LJ_E,double *LJ_S,double *Box,double cutoff,double *Pt_ePos,double spr_k,double *Mass);

	Box[0][0]=0;
  Box[0][1]=Pt_I*fcc_lattice;
  Box[1][0]=0;
  Box[1][1]=Pt_J*fcc_lattice;
  Box[2][0]=-(Pt_K-0.5)*fcc_lattice;
  Box[2][1]=d;
  cout<<"计算区域X: "<<Box[0][0]<<", "<<Box[0][1]<<"\n";
	cout<<"计算区域Y: "<<Box[1][0]<<", "<<Box[1][1]<<"\n";
	cout<<"计算区域Z: "<<Box[2][0]<<", "<<Box[2][1]<<"\n";
	//位置，速度初始化
	cudaMalloc((void**)&d_All_type,sizeof(All_type));
	cudaMalloc((void**)&d_All_Pos,sizeof(All_Pos));
	cudaMalloc((void**)&d_All_Vel,sizeof(All_Vel));
	cudaMalloc((void**)&d_All_Acc,sizeof(All_Acc));
	cudaMemcpy(d_All_type,All_type,sizeof(All_type),cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Pos,All_Pos,sizeof(All_Pos),cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel,All_Vel,sizeof(All_Vel),cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc,All_Acc,sizeof(All_Acc),cudaMemcpyHostToDevice);
	srand((unsigned)time(NULL));
	Initialization_Kernel<<<1,Pt_N+Ar_N>>>(d_All_type,d_All_Pos,d_All_Vel,fcc_lattice,mp_V,Box,PI);
  //首次位置周期
  *temp=nan;
  Pos_period<<<1,Pt_N+Ar_N>>>(d_All_Pos,Box,Pt_ePos,temp);
	Box[2][0]=*temp;
  //首次控温
  Pt_argVel={0.0,0.0,0.0};
  *Pt_V2=0.0;
  *Pt_T=0.0;
  rescale_T<<<1,Pt_N>>>(d_All_Vel,Pt_argVel,Pt_V2,Pt_T,nd_Velocity,Mass,nd_Mass,kB,T);
  //首次加速度周期
  Acceleration_period<<<1,Pt_N+Ar_N>>>(d_All_Pos,d_All_Acc,LJ_E,LJ_S,Box,cutoff,Pt_ePos,spr_k,Mass);
	cudaMemcpy(All_type,d_All_type,sizeof(All_type),cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Pos,d_All_Pos,sizeof(All_Pos),cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Vel,d_All_Vel,sizeof(All_Vel),cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Acc,d_All_Acc,sizeof(All_Acc),cudaMemcpyDeviceToHost);
	cudaFree(d_All_type);
	cudaFree(d_All_Pos);
	cudaFree(d_All_Vel);
	cudaFree(d_All_Acc);
  Pt_ePos=All_Pos[:Pt_N][:];
  //初始信息
  cout<<"Created "<<Pt_N<<" Pt\n";
	cout<<"Created "<<Ar_N<<" Ar\n";
	cout<<"Pt整体x方向平均速度"<<Pt_argVel[0]<<"\n";
	cout<<"Pt整体y方向平均速度"<<Pt_argVel[1]<<"\n";
	cout<<"Pt整体z方向平均速度"<<Pt_argVel[2]<<"\n";
	cout<<"Pt温度"<<Pt_T<<"\n";
	cout<<"Ar入射速度:"<<All_Vel[Pt_N][0]<<","<<All_Vel[Pt_N][1]<<","<<All_Vel[Pt_N][2]<<"\n";
	cout<<"*******Model Initialization Done!*******\n";
}

__global__ void Initialization_Kernel(int *All_type,double *All_Pos,double *All_Vel,double fcc_lattice,double *mp_V,double *Box,double PI){
	int i,jk,j,k;
	double R1,R2;
	int tid=threadIdx.x;

	if(tid<Pt_N){//Pt
		i=int(tid/(2*Pt_J*Pt_K));
		jk=tid%(2*Pt_J*Pt_K);
		j=int(jk/Pt_K);
		k=jk%Pt_K;
		if(((i%2)+(j%2))%2==0){
			k=2*k;
		}
		else{
			k=2*k+1;
		}
		All_type[tid]=1;
		All_Pos[tid][0]=i/2*fcc_lattice;
		All_Pos[tid][1]=j/2*fcc_lattice;
		All_Pos[tid][2]=(k/2-2.5)*fcc_lattice;
		for(axis=0;axis<3;axis++){
			R1 = 0.;
			while (R1 == 0.) {
				R1 = rand() / double(RAND_MAX);
			}
			R2 = 0.;
			while (R2 == 0.) {
				R2 = rand() / double(RAND_MAX);
			}
			All_Vel[tid][dim] = mp_V[1] / sqrt(3)*sqrt(-2 * log(R1))*cos(2 * PI*R2);//高斯分布，平均值为0，方差=方均根速度用来控温
		}
	}
	if(tid==Pt_N){//Ar
		All_type[tid]=0;
		R1 = 0.;
		while (R1 == 0.) {
			R1 = rand() / double(RAND_MAX);
		}
		R2 = 0.;
		while (R2 == 0.) {
			R2 = rand() / double(RAND_MAX);
		}
		All_Pos[tid][0]=Box[0][0]+(Box[0][1]-Box[0][0])*R1;
		All_Pos[tid][1]=Box[1][0]+(Box[1][1]-Box[1][0])*R2;
		All_Pos[tid][2]=Box[2][1];
		R1 = 0.;
		while (R1 == 0.) {
			R1 = rand() / double(RAND_MAX);
		}
		R2 = 0.;
		while (R2 == 0.) {
			R2 = rand() / double(RAND_MAX);
		}
		All_Vel[tid][0]=mp_V[0]*sqrt(-log(R1))*cos(2*PI*R2);//Maxwell分布
		R1 = 0.;
		while (R1 == 0.) {
			R1 = rand() / double(RAND_MAX);
		}
		R2 = 0.;
		while (R2 == 0.) {
			R2 = rand() / double(RAND_MAX);
		}
		All_Vel[tid][1]=mp_V[0]*sqrt(-log(R1))*sin(2*pi*R2);
		R1 = 0.;
		while (R1 == 0.) {
			R1 = rand() / double(RAND_MAX);
		}
		All_Vel[tid][2]=-mp_V[0]*sqrt(-log(R1));
	}
}

/******************************************************************************/
__global__ void Pos_period(double *All_Pos,double *Box,double *Pt_ePos,double *temp) {
	int axis, i;
	int tid=threadIdx.x;

	if(tid<Pt.N+Ar.N){
		//X,Y方向周期
		for axis in range(2){
			if(All_Pos[tid][axis]<Box[axis][0]){
				All_Pos[tid][axis]+=Box[axis][1]-Box[axis][0];
				if(tid<Pt.N){
					Pt_ePos[tid][axis]+=Box[axis][1]-Box[axis][0];
				}
			}
			else if(All_Pos[tid][axis]>=Box[axis][1]){
				All_Pos[tid][axis]-=Box[axis][1]-Box[axis][0];
				if(tid<Pt.N){
					Pt_ePos[tid][axis]-=Box[axis][1]-Box[axis][0];
				}
			}
		}
		//Z方向下边界更新
		atomicMin(&temp,All_Pos[tid][2]);
	}
}

/******************************************************************************/
__global__ void rescale_T(double *All_Vel,double *Pt_argVel,double *Pt_V2,double *Pt_T,double nd_Velocity,double *Mass,double nd_Mass,double kB,double *T){
	int tid=threadIdx.x;

	if(tid<Pt_N){
		atomicAdd(&Pt_argVel[0],(All_Vel[tid][0])/Pt_N);
		atomicAdd(&Pt_argVel[1],(All_Vel[tid][1])/Pt_N);
		atomicAdd(&Pt_argVel[2],(All_Vel[tid][2])/Pt_N);
		__syncthreads();
		//只需要热运动速度
		All_Vel[tid][0]-=Pt_argVel[0];
		All_Vel[tid][1]-=Pt_argVel[1];
		All_Vel[tid][2]-=Pt_argVel[2];
		atomicAdd(&Pt_V2,All_Vel[tid][0]**2+All_Vel[tid][1]**2+All_Vel[tid][2]**2);
		__syncthreads();
		Pt_T=Pt_V2*nd_Velocity**2*Mass[1]*nd_Mass/(3*Pt_N*kB);
		All_Vel[tid][0]*=sqrt(T[1]/Pt_T);
		All_Vel[tid][1]*=sqrt(T[1]/Pt_T);
		All_Vel[tid][2]*=sqrt(T[1]/Pt_T);
	}
}

/******************************************************************************/
__global__ void Acceleration_period(double *All_Pos,double *All_Acc,double *LJ_E,double *LJ_S,double *Box,double cutoff,double *Pt_ePos,double spr_k,double *Mass) {
	int i, LJ_pair;
	double Epair, Spair, Pairx, Pairy, Pairz, Dispair, Fpair, Atom_Fx, Atom_Fy, Atom_Fz;
	double Spring_Disx, Spring_Fx, Pt_Fx, Spring_Disy, Spring_Fy, Pt_Fy, Spring_Disz, Spring_Fz, Pt_Fz, Ar_Fx, Ar_Fy, Ar_Fz;
	int tid=threadIdx.x;

	if(tid<Pt_N+Ar_N){
		Atom_Fx=0.;
		Atom_Fy=0.;
		Atom_Fz=0.;
		for(i=0;i<Pt_N+Ar_N;i++){
			if(tid<Pt_N && i<Pt_N){
				LJ_pair=1;
			}
			else if(tid>=Pt_N && i>=Pt_N){
				LJ_pair=0;
			}
			else{
				LJ_pair=2;
			}
			Epair=LJ_E[LJ_pair];
			Spair=LJ_S[LJ_pair];
			//周期相对位置
			Pairx=All_Pos[tid][0]-All_Pos[i][0];
			Pairy=All_Pos[tid][1]-All_Pos[i][1];
			Pairz=All_Pos[tid][2]-All_Pos[i][2];
			if(abs(Pairx)>=Box[0][1]-Box[0][0]-cutoff){
				Pairx-=(Box[0][1]-Box[0][0])*Pairx/abs(Pairx);
			}
			if(abs(Pairy)>=Box[1][1]-Box[1][0]-cutoff){
				Pairy-=(Box[1][1]-Box[1][0])*Pairy/abs(Pairy);
			}
			//周期距离
			Dispair=sqrt(Pairx**2+Pairy**2+Pairz**2);
			if(Dispair>0 and Dispair<=cutoff):
					Fpair=48*Epair*(Spair**12/Dispair**13-0.5*Spair**6/Dispair**7);
					Atom_Fx+=Pairx*Fpair/Dispair;
					Atom_Fy+=Pairy*Fpair/Dispair;
					Atom_Fz+=Pairz*Fpair/Dispair;
		}
		if(tid<Pt_N){
			//Pt弹性恢复力
			Spring_Disx=All_Pos[tid][0]-Pt_ePos[tid][0];
			Spring_Fx=-spr_k*Spring_Disx;
			Pt_Fx=Atom_Fx+Spring_Fx;
			All_Acc[tid][0]=Pt_Fx/Mass[1];
			Spring_Disy=All_Pos[tid][1]-Pt_ePos[tid][1];
			Spring_Fy=-spr_k*Spring_Disy;
			Pt_Fy=Atom_Fy+Spring_Fy;
			All_Acc[tid][1]=Pt_Fy/Mass[1];
			Spring_Disz=All_Pos[tid][2]-Pt_ePos[tid][2];
			Spring_Fz=-spr_k*Spring_Disz;
			Pt_Fz=Atom_Fz+Spring_Fz;
			All_Acc[tid][2]=Pt_Fz/Mass[1];
		}
		else{
			Ar_Fx=Atom_Fx;
			All_Acc[tid][0]=Ar_Fx/Mass[0];
			Ar_Fy=Atom_Fy;
			All_Acc[tid][1]=Ar_Fy/Mass[0];
			Ar_Fz=Atom_Fz;
			All_Acc[tid][2]=Ar_Fz/Mass[0];
		}
	}
}

/******************************************************************************/
__global__ void Verlet_Pos(double *All_Pos,double *All_Vel,double *All_Acc,double dt){
	int axis;
	int tid=threadIdx.x;

	if(tid<Pt_N+Ar_N){
		for(axis=0;axis<3;axis++){
			All_Pos[tid][axis]+=All_Vel[tid][axis]*dt+0.5*All_Acc[tid][axis]*dt**2;
		}
	}
}

/******************************************************************************/
__global__ void Verlet_Vel(double *All_Vel,double *All_Acc_temp,double *All_Acc,double dt){
	int axis;
	int tid=threadIdx.x;

	if(tid<Pt_N+Ar_N){
		for(axis=0;axis<3;axis++){
			All_Vel[tid][axis]+=0.5*(All_Acc_temp[tid][axis]+All_Acc[tid][axis])*dt;
		}
	}
}

/******************************************************************************/
void Parameters::Dump(int All_type[], double All_Pos[][], int timestep, int ds) {
	int i;

	if (timestep%ds == 0) {
		ofstream MD;
		MD.open("Kernel_MD_CUDA_C.dump", ios::app);
		MD << "ITEM: TIMESTEP\n";
		MD << timestep << "\n";
		MD << "ITEM: NUMBER OF ATOMS\n";
		MD << Pt_N + Ar_N << "\n";
		MD << "ITEM: BOX BOUNDS pp pp ff\n";
		MD << Box[0][0] << " " << Box[0][1] << "\n";
		MD << Box[1][0] << " " << Box[1][1] << "\n";
		MD << Box[2][0] << " " << Box[2][1] << "\n";
		MD << "ITEM: ATOMS id type x y z\n";
		for (i = 0;i < Pt_N_Ar_N;i++) {
			MD << i + 1 << " " << All_type[i] + 1 << " " << All_Pos[i][0] << " " << All_Pos[i][1] << " " << All_Pos[i][2] << "\n";
		}
		MD.close();
		ofstream Zt;
		Zt.open("Kernel_MD_CUDA_C_Zt.dat", ios::app);
		Zt << timestep * dt << " " << All_Pos[Pt_N][2] << "\n";
		Zt.close();
	}
}

/******************************************************************************/
void Parameters::Exit(int All_type[], double All_Pos[][], int timestep) {

	if (All_Pos[Pt_N][2] > d || timestep >= Tt) {
		state = false;
		Dump(All_type, All_Pos, timestep);
	}
	else {
		Dump(All_type, All_Pos, timestep, dumpstep);
	}
}

////////////////////////////////////////////////////////////////////////////////
/*************************************main*************************************/
////////////////////////////////////////////////////////////////////////////////
int main() {
	class Parameters Pars;
	clock_t start, finish;
	double tperl;
	int All_type[Pt_N+Ar_N];
	double All_Pos[Pt_N+Ar_N][3], All_Vel[v][3], All_Acc[Pt_N+Ar_N][3];
	double *d_All_Acc,*d_All_Vel,*d_All_Acc,*d_All_Acc_temp;
	int timestep;
	double *temp,Pt_argVel[3],*Pt_V2,*Pt_T;
	__global__ void Verlet_Pos(double *All_Pos,double *All_Vel,double *All_Acc,double dt);
	__global__ void Pos_period(double *All_Pos,double *Box,double *Pt_ePos,double *temp);
	__global__ void Acceleration_period(double *All_Pos,double *All_Acc,double *LJ_E,double *LJ_S,double *Box,double cutoff,double *Pt_ePos,double spr_k,double *Mass);
	__global__ void Verlet_Vel(double *All_Vel,double *All_Acc_temp,double *All_Acc,double dt);
	__global__ void rescale_T(double *All_Vel,double *Pt_argVel,double *Pt_V2,double *Pt_T,double nd_Velocity,double *Mass,double nd_Mass,double kB,double *T);

	Pars.Init();
	Pars.Initialization(All_type, All_Pos, All_Vel, All_Acc);
	timestep=0;
	Pars.Dump(All_type, All_Pos, timestep);
	cudaMalloc((void**)&d_All_Pos,sizeof(All_Pos));
	cudaMalloc((void**)&d_All_Vel,sizeof(All_Vel));
	cudaMalloc((void**)&d_All_Acc,sizeof(All_Acc));
	cudaMalloc((void**)&d_All_Acc_temp,sizeof(All_Acc));
	cudaMemcpy(d_All_Pos,All_Pos,sizeof(All_Pos),cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel,All_Vel,sizeof(All_Vel),cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc,All_Acc,sizeof(All_Acc),cudaMemcpyHostToDevice);
	start = clock();
	while (Pars.state) {
	  Verlet_Pos<<<1,Pt_N+Ar_N>>>(d_All_Pos,d_All_Vel,d_All_Acc,Pars.dt);
	  *temp=nan;
	  Pos_period<<<1,Pt_N+Ar_N>>>(d_All_Pos,Pars.Box,Pars.Pt_ePos,temp);
		Box[2][0]=*temp;
		d_All_Acc_temp=d_All_Acc;
	  Acceleration_period<<<1,Pt_N+Ar_N>>>(d_All_Pos,d_All_Acc,Pars.LJ_E,Pars.LJ_S,Pars.Box,Pars.cutoff,Pars.Pt_ePos,Pars.spr_k,Pars.Mass);
	  Verlet_Vel<<<1,Pt_N+Ar_N>>>(d_All_Vel,d_All_Acc_temp,d_All_Acc,Pars.dt);
		Pt_argVel={0.0,0.0,0.0};
	  *Pt_V2=0.0;
	  *Pt_T=0.0;
	  rescale_T<<<1,Pt_N>>>(d_All_Vel,Pt_argVel,Pt_V2,Pt_T,Pars.nd_Velocity,Pars.Mass,Pars.nd_Mass,Pars.kB,Pars.T);
	  timestep+=1;
		cudaMemcpy(All_Pos,d_All_Pos,sizeof(All_Pos),cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Vel,d_All_Vel,sizeof(All_Vel),cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Acc,d_All_Acc,sizeof(All_Acc),cudaMemcpyDeviceToHost);
	  Pars.Exit(All_type, All_Pos, timestep)
		finish = clock();
		tperl = double(finish - start) / CLOCKS_PER_SEC / timestep;
		cout << timestep << " TimeSteps; ArgTime: " << tperl << " Seconds!\r";
	}
	cudaFree(d_All_Pos);
	cudaFree(d_All_Vel);
	cudaFree(d_All_Acc);
	cudaFree(d_All_Acc_temp);
	system("pause");
	return 0;
}
