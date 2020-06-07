#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include<algorithm>
using namespace std;
const int Pt_I = 6, Pt_J = 6, Pt_K = 3;
const int Lib_N=100000;
const int SampleParallel_N=1000;

struct Atom{
	double x;
	double y;
	double z;
	double vx;
	double vy;
	double vz;
	double ax;
	double ay;
	double az;
}

struct Atom_State{
	int N;
	double m;
	double LJ_E;
	double LJ_S;
	double T;
	double mp_V;
	double fcc_lattice;
}

struct ND_Pars{
	double Mass;
	double Energy;
	double Length;
	double Velocity;
	double Time;
	double Acceleration;
}

class Parameters {
public:
	Atom Pt[4*Pt_I*Pt_J*Pt_K]
	Atom Pt_Init[4*Pt_I*Pt_J*Pt_K];
	Atom ArLib[Lib_N];
	Atom Ar[SampleParallel_N];
	Atom_State Pt_State;
	Atom_State Ar_State;
	ND_Pars Pars;

	int Rt, Tt, dumpstep, Pt_N, Ar_N, All_N, Block_N;
	double PI, kB, ArPt_LJ_E, ArPt_LJ_S, cutoff, d, spr_k, dt, Pt_V2, Pt_T;
	double Box[3][2], Pt_argVel[3];
	bool state;
	void Init();
	void InitializationLib(double *ArLib_Pos,double *ArLib_Vel);
	void InitializationPt(double *Pt_Pos,double *Pt_Vel,double *Pt_Acc);
	void Pos_period(double Ar_Pos[][3], double Pt_Pos[][3]);
	void Acceleration_period(double Ar_Pos[][3], double Pt_Pos[][3], double Ar_Acc[][3], double Pt_Acc[][3]);
	void Verlet(double Ar_Pos[][3], double Ar_Vel[][3], double Ar_Acc[][3], double Pt_Pos[][3], double Pt_Vel[][3], double Pt_Acc[][3]);
	void rescale_T(double Pt_Vel[][3]);
	void Dump(double Pt_Pos[][3], double Ar_Pos[][3], int timestep, int ds = 1);
	void Exit(double Pt_Pos[][3], double Ar_Pos[][3], int timestep);
	double random();
};

void Parameters::Init() {
	//物理参数
	PI = 3.14159265;
	kB = 1.38E-23;
	//原子参数
	Ar_State.N = 0;
	Ar_State.m = 39.95 / 6.02 * 1E-26;//单位kg
	Ar_State.LJ_E = 1.654E-21;//单位J
	Ar_State.LJ_S = 3.40 * 1E-10;//单位m
	Ar_State.T = 300.;//单位K
	Ar_State.mp_V = sqrt(2 * kB*T[0] / Mass[0]);//气体最概然速率
	Pt_State.N = 0;
	Pt_State.m = 195.08 / 6.02 * 1E-26;
	Pt_State.LJ_E = 5.207E-20;
	Pt_State.LJ_S = 2.47 * 1E-10;
	Pt_State.T = 300.;
	Pt_State.mp_V = sqrt(3 * kB*T[1] / Mass[1]);//固体方均根速率
	Pt_State.fcc_lattice = 3.93E-10;
	ArPt_LJ_E = 1.093E-21;
	ArPt_LJ_S = 2.94 * 1E-10;
	//无量纲参数
	Pars.Mass = Pt_State.m;
	Pars.Energy = Pt_State.LJ_E;
	Pars.Length = Pt_State.LJ_S;
	Pars.Velocity = sqrt(Pars.Energy / Pars.Mass);
	Pars.Time = Pars.Length / Pars.Velocity;
	Pars.Acceleration = Pars.Energy / (Pars.Mass * Pars.Length);
	//无量纲化
	Ar_State.m /= Pars.Mass;
	Ar_State.LJ_E /= Pars.Energy;
	Ar_State.LJ_S /= Pars.Length;
	Ar_State.mp_V /= Pars.Velocity;
	Pt_State.m /= Pars.Mass;
	Pt_State.LJ_E /= Pars.Energy;
	Pt_State.LJ_S /= Pars.Length;
	Pt_State.mp_V /= Pars.Velocity;
	Pt_State.fcc_lattice /= Pars.Velocity;
	ArPt_LJ_E /= Pars.Energy;
	ArPt_LJ_S /= Pars.Velocity;
	cutoff = 10 * 1E-10 / Pars.Length;
	//模拟参数
	d = 5.0;
	spr_k = 5000.;
	dt = 0.001;
	Rt = 100;
	Tt = 3500000;
	dumpstep = 100;
	All_N=Pt_State.N+Ar_State.N;
	Pt_argVel={0.0,0.0,0.0};
	Pt_V2=0.0;
	Pt_T=0.0;
	state = true;
	Box[0][0]=nan;
	Box[1][0]=nan;
	Box[2][0]=nan;
	Box[0][1]=nan;
	Box[1][1]=nan;
	Box[2][1]=d;
	Block_N=1;

	cout << "Parameters Initialized!\n";
}

/******************************************************************************/
void Parameters::InitializationPt(){
	int i;
	double *d_Pt,*d_Ar;
	__global__ void InitializationPt_Kernel(Atom *Pt,Atom_State Pt_State,double PI);
	__global__ void InitializationBox_Kernel(Atom *Pt,Atom_State Pt_State,double *Box);
	__global__ void rescale_T(Atom *Pt,Atom_State Pt_State,ND_Pars Pars,double *Pt_argVel,double Pt_V2,double Pt_T,double kB);
	__global__ void Acceleration_period(Atom *Pt, Atom *Ar, Atom_State Pt_State, Atom_State Ar_State,double *Box,double cutoff,double spr_k);

	seedt=time(NULL);
  srand((unsigned)seedt);
  //位置，速度初始化
	cudaMalloc((void**)&d_Pt,sizeof(Pt));
	cudaMalloc((void**)&d_Pt_Init,sizeof(Pt));
	cudaMalloc((void**)&d_Ar,sizeof(Ar));
  InitializationPt_Kernel<<<Block_N,4*Pt_I*Pt_J*Pt_K>>>(d_Pt,Pt_State,PI);
	cudaDeviceSynchronize();
	cudaMemcpy(Pt_Init,d_Pt,sizeof(Pt),cudaMemcpyDeviceToHost);
	cudaMemcpy(d_Pt_Init,Pt_Init,sizeof(Ar),cudaMemcpyHostToDevice);
	//建立盒子
	InitializationBox_Kernel<<<Block_N,Pt_State.N>>>(d_Pt,Pt_State,Box);
	cudaDeviceSynchronize();
  //首次控温
  rescale_T<<<Block_N,Pt_State.N>>>(d_Pt,Pt_State,Pars,Pt_argVel,Pt_V2,Pt_T,kB);
	cudaDeviceSynchronize();
  //首次加速度周期
  Acceleration_period<<<Block_N,Pt_State.N+Ar_State.N>>>(d_Pt,d_Ar,Pt_State,Ar_State,Box,cutoff,spr_k);
	cudaDeviceSynchronize();
	//初始信息
  cout<<"Created"<<Pt_State.N<<"Pt\n";
	cout<<"Pt整体x方向平均速度"<<Pt_argVel[0]<<"\n";
	cout<<"Pt整体y方向平均速度"<<Pt_argVel[1]<<"\n";
	cout<<"Pt整体z方向平均速度"<<Pt_argVel[2]<<"\n";
	cout<<"Pt温度"<<Pt_T<<"\n";
  cout<<"*******Model-Pt Initialization Done!*******\n";
	//弛豫过程
	for(i=0;i<Rt;i++){
		Verlet();
		Pt_argVel={0.0,0.0,0.0};
	  Pt_V2=0.0;
	  Pt_T=0.0;
	  rescale_T<<<Block_N,Pt_State.N>>>(d_Pt,Pt_State,Pars,Pt_argVel,Pt_V2,Pt_T,kB);
		cudaDeviceSynchronize();
	}
	cout<<"弛豫"<<Rt<<"个时间步\n";
	cudaMemcpy(Pt,d_Pt,sizeof(Pt),cudaMemcpyDeviceToHost);
	cudaMemcpy(Pt_Init,d_Pt,sizeof(Pt),cudaMemcpyDeviceToHost);
	cudaMemcpy(Ar,d_Ar,sizeof(Ar),cudaMemcpyDeviceToHost);
	cudaFree(d_Pt);
	cudaFree(d_Pt_Init);
	cudaFree(d_Ar);

}

/******************************************************************************/
__global__ void InitializationPt_Kernel(Atom *Pt,Atom_State Pt_State,double PI){
	int i,jk,j,k,axis;
	double R1,R2;
	int tid=threadIdx.x+blockIdx.x*blockDim.x;

	if(tid<4*Pt_I*Pt_J*Pt_K){
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
		Pt[tid].x=i/2*Pt_State.fcc_lattice;
		Pt[tid].y=j/2*Pt_State.fcc_lattice;
		Pt[tid].z=(k/2-2.5)*Pt_State.fcc_lattice;
		R1 = random();
		R2 = random();
		Pt[tid].vx=Pt_State.mp_V/sqrt(3.0)*sqrt(-2*log(R1))*cos(2*PI*R2);//高斯分布，平均值为0，方差=方均根速度用来控温
		R1 = random();
		R2 = random();
		Pt[tid].vy=Pt_State.mp_V/sqrt(3.0)*sqrt(-2*log(R1))*cos(2*PI*R2);
		R1 = random();
		R2 = random();
		Pt[tid].vz=Pt_State.mp_V/sqrt(3.0)*sqrt(-2*log(R1))*cos(2*PI*R2);
		atomicAdd(&Pt_State.N,1);
	}
}

/******************************************************************************/
__global__ void InitializationBox_Kernel(Atom *Pt,Atom_State Pt_State,double *Box){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;

	if(tid<Pt_State.N){
		atomicMin(&Box[0][0],Pt[tid].x);
		atomicMin(&Box[1][0],Pt[tid].y);
		atomicMin(&Box[2][0],Pt[tid].z);
		atomicMax(&Box[0][1],Pt[tid].x+Pt_State.fcc_lattice/2);
		atomicMax(&Box[1][1],Pt[tid].y+Pt_State.fcc_lattice/2);
	}
}

/******************************************************************************/
__global__ void rescale_T(Atom *Pt,Atom_State Pt_State,ND_Pars Pars,double *Pt_argVel,double Pt_V2,double Pt_T,double kB){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double Pt_T;

  if(tid<Pt_State.N){
		atomicAdd(&(Pt_argVel[0]),(Pt[tid].vx)/Pt_State.N);
		atomicAdd(&(Pt_argVel[1]),(Pt[tid].vy)/Pt_State.N);
		atomicAdd(&(Pt_argVel[2]),(Pt[tid].vz)/Pt_State.N);
		__syncthreads();
		//只需要热运动速度
		Pt[tid].vx-=Pt_argVel[0];
		Pt[tid].vy-=Pt_argVel[1];
		Pt[tid].vz-=Pt_argVel[2];
		atomicAdd(&Pt_V2,Pt[tid].vx**2+Pt[tid].vy**2+Pt[tid].vz**2);
		__syncthreads();
		Pt_T=Pt_V2*Pars.Velocity**2*Pt_State.m*Pars.Mass/(3*Pt_State.N*kB);
		Pt[tid].vx*=sqrt(Pt_State.T/Pt_T);
		Pt[tid].vy*=sqrt(Pt_State.T/Pt_T);
		Pt[tid].vz*=sqrt(Pt_State.T/Pt_T);
	}
}

/******************************************************************************/
__global__ void Acceleration_period(Atom *Pt, Atom *Ar, Atom_State Pt_State, Atom_State Ar_State,double *Box,double cutoff,double spr_k) {
	double Atom_Fx,Atom_Fy,Atom_Fz,Epair,Spair,Pairx,Pairy,Pairz,Dispair,Spring_Disx,Spring_Disy,Spring_Disz,Spring_Fx,Spring_Fy,Spring_Fz,Pt_Fx,Pt_Fy,Pt_Fz,Ar_Fx,Ar_Fy,Ar_Fz;
	int i;
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int bid=blockIdx.x;


  if(tid<Pt_State.N+Ar_State.N){
		Atom_Fx=0.0;
		Atom_Fy=0.0;
		Atom_Fz=0.0;
		//与Pt相互作用
		for(i=0;i<Pt_State.N+Ar_State.N;i++){
			if(tid<Pt_State.N && i<Pt_State.N){
				Epair=Pt_State.LJ_E;
				Spair=Pt_State.LJ_S;
				Pairx=Pt[tid].x-Pt[i].x;
				Pairy=Pt[tid].y-Pt[i].y;
				Pairz=Pt[tid].z-Pt[i].z;
			}
			else if(tid<Pt_State.N && i>=Pt_State.N){
				Epair=ArPt_LJ_E;
				Spair=ArPt_LJ_S;
				Pairx=Pt[tid].x-Ar[bid].x;
				Pairy=Pt[tid].y-Ar[bid].y;
				Pairz=Pt[tid].z-Ar[bid].z;
			}
			else if(tid>=Pt_State.N && i<Pt_State.N){
				Epair=ArPt_LJ_E;
				Spair=ArPt_LJ_S;
				Pairx=Ar[bid].x-Pt[i].x;
				Pairy=Ar[bid].y-Pt[i].y;
				Pairz=Ar[bid].z-Pt[i].z;
			}
			else{
				Epair=Ar_State.LJ_E;
				Spair=Ar_State.LJ_S;
				Pairx=0.0;
				Pairy=0.0;
				Pairz=0.0;
			}
			//周期相对位置
			if(abs(Pairx)>=Box[0][1]-Box[0][0]-cutoff){
				Pairx-=(Box[0][1]-Box[0][0])*Pairx/abs(Pairx);
			}
			if(abs(Pairy)>=Box[1][1]-Box[1][0]-cutoff){
				Pairy-=(Box[1][1]-Box[1][0])*Pairy/abs(Pairy);
			}
			//周期距离
			Dispair=sqrt(Pairx**2+Pairy**2+Pairz**2);
			if(Dispair>0 && Dispair<=cutoff){
				Fpair=48*Epair*(Spair**12/Dispair**13-0.5*Spair**6/Dispair**7);
				Atom_Fx+=Pairx*Fpair/Dispair;
				Atom_Fy+=Pairy*Fpair/Dispair;
				Atom_Fz+=Pairz*Fpair/Dispair;
			}
		}
		if(tid<Pt_State.N){
			//Pt弹性恢复力
			Spring_Disx=Pt[tid].x-Pt_Init[tid].x;
			Spring_Disy=Pt[tid].y-Pt_Init[tid].y;
			Spring_Disz=Pt[tid].z-Pt_Init[tid].z;
			Spring_Fx=-spr_k*Spring_Disx;
			Spring_Fy=-spr_k*Spring_Disy;
			Spring_Fz=-spr_k*Spring_Disz;
			Pt_Fx=Atom_Fx+Spring_Fx;
			Pt_Fy=Atom_Fy+Spring_Fy;
			Pt_Fz=Atom_Fz+Spring_Fz;
			Pt[tid].ax=Pt_Fx/Pt_State.m;
			Pt[tid].ay=Pt_Fy/Pt_State.m;
			Pt[tid].az=Pt_Fz/Pt_State.m;
		}
		else{
			Ar_Fx=Atom_Fx;
			Ar_Fy=Atom_Fy;
			Ar_Fz=Atom_Fz;
			Ar[bid].ax=Ar_Fx/Ar_State.m;
			Ar[bid].ay=Ar_Fy/Ar_State.m;
			Ar[bid].az=Ar_Fz/Ar_State.m;
		}
	}
}

/******************************************************************************/
void Parameters::Verlet() {
	Atom *d_Pt_temp,*d_Ar_temp;
	__global__ void Verlet_Pos(Atom *Pt,Atom *Ar,Atom_State Pt_State,Atom_State Ar_State,double dt);
	__global__ void Pos_period(Atom *Pt,Atom *Pt_Init,Atom *Ar,Atom_State Pt_State,Atom_State Ar_State,double *Box);
	__global__ void Acceleration_period(Atom *Pt, Atom *Ar, Atom_State Pt_State, Atom_State Ar_State,double *Box,double cutoff,double spr_k);
	__global__ void Verlet_Vel(Atom *Pt,Atom *Pt_temp,Atom *Ar,Atom *Ar_temp,Atom_State Pt_State,Atom_State Ar_State,double dt);

  Verlet_Pos<<<Block_N,Pt_State.N+Ar_State.N>>>(d_Pt,d_Ar,Pt_State,Ar_State,dt);
	cudaDeviceSynchronize();
  Pos_period<<<Block_N,Pt_State.N+Ar_State.N>>>(d_Pt,d_Pt_Init,d_Ar,Pt_State,Ar_State,Box);
	cudaDeviceSynchronize();
  d_Pt_temp=d_Pt;
	d_Ar_temp=d_Ar;
  Acceleration_period<<<Block_N,Pt_State.N+Ar_State.N>>>(d_Pt, d_Ar, Pt_State, Ar_State,Box,cutoff,spr_k);
	cudaDeviceSynchronize();
  Verlet_Vel<<<Block_N,Pt_State.N+Ar_State.N>>>(d_Pt,d_Pt_temp,d_Ar,d_Ar_temp,Pt_State,Ar_State,dt);
	cudaDeviceSynchronize();
}

/******************************************************************************/
__global__ void Verlet_Pos(Atom *Pt,Atom *Ar,Atom_State Pt_State,Atom_State Ar_State){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int bid=blockIdx.x;

  if(tid<Pt_State.N+Ar_State.N){
		if(tid<Pt_State.N){
			Pt[tid].x+=Pt[tid].vx*dt+0.5*Pt[tid].ax*dt**2;
			Pt[tid].y+=Pt[tid].vy*dt+0.5*Pt[tid].ay*dt**2;
			Pt[tid].z+=Pt[tid].vz*dt+0.5*Pt[tid].az*dt**2;
		}
		else{
			Ar[bid].x+=Ar[bid].vx*dt+0.5*Ar[bid].ax*dt**2;
			Ar[bid].y+=Ar[bid].vy*dt+0.5*Ar[bid].ay*dt**2;
			Ar[bid].z+=Ar[bid].vz*dt+0.5*Ar[bid].az*dt**2;
		}
	}
}
/******************************************************************************/
__global__ void Pos_period(Atom *Pt,Atom *Pt_Init,Atom *Ar,Atom_State Pt_State,Atom_State Ar_State,double *Box) {
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int bid=blockIdx.x;

  if(tid<Pt_State.N+Ar_State.N){
		if(tid<Pt_State.N){
			//X方向
			if(Pt[tid].x<Box[0][0]){
				Pt[tid].x+=Box[0][1]-Box[0][0];
				Pt_Init[tid].x+=Box[0][1]-Box[0][0];
			}
			else if(Pt[tid].x>=Box[0][1]){
				Pt[tid].x-=Box[0][1]-Box[0][0];
				Pt_Init[tid].x-=Box[0][1]-Box[0][0];
			}
			//Y方向
			if(Pt[tid].y<Box[1][0]){
				Pt[tid].y+=Box[1][1]-Box[1][0];
				Pt_Init[tid].y+=Box[1][1]-Box[1][0];
			}
			else if(Pt[tid].y>=Box[1][1]){
				Pt[tid].y-=Box[1][1]-Box[1][0];
				Pt_Init[tid].y-=Box[1][1]-Box[1][0];
			//Z方向下边界更新
			atomicMin(&Box[2][0],Pt[tid].z);
			}
		}
		else{
			//X方向
			if(Ar[bid].x<Box[0][0]){
				Ar[bid].x+=Box[0][1]-Box[0][0];
			}
			else if(Ar[bid].x>=Box[0][1]){
				Ar[bid].x-=Box[0][1]-Box[0][0];
			}
			//Y方向
			if(Ar[bid].y<Box[1][0]){
				Ar[bid].y+=Box[1][1]-Box[1][0];
			}
			else if(Ar[bid].y>=Box[1][1]){
				Ar[bid].y-=Box[1][1]-Box[1][0];
			}
		}
	}
}

/******************************************************************************/
__global__ void Verlet_Vel(Atom *Pt,Atom *Pt_temp,Atom *Ar,Atom *Ar_temp,Atom_State Pt_State,Atom_State Ar_State,double dt){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int bid=blockIdx.x;

  if(tid<Pt_State.N+Ar_State.N){
		if(tid<Pt_State.N){
			Pt[tid].vx+=0.5*(Pt_temp[tid].ax+Pt[tid].ax)*dt;
			Pt[tid].vy+=0.5*(Pt_temp[tid].ay+Pt[tid].ay)*dt;
			Pt[tid].vz+=0.5*(Pt_temp[tid].az+Pt[tid].az)*dt;
		}
		else{
			Ar[bid].vx+=0.5*(Ar_temp[bid].ax+Ar[bid].ax)*dt;
			Ar[bid].vy+=0.5*(Ar_temp[bid].ay+Ar[bid].ay)*dt;
			Ar[bid].vz+=0.5*(Ar_temp[bid].az+Ar[bid].az)*dt;
		}
	}
}
/******************************************************************************/
void Parameters::InitializationLib(){
	double *d_ArLib;
	int seedt;
	__global__ void InitializationLib_Kernel(Atom *ArLib,Atom_State Ar_State,double *Box,double PI);

	cudaMalloc((void**)&d_ArLib,sizeof(ArLib));
	seedt+=1;
  srand((unsigned)seedt);
  InitializationLib_Kernel<<<(Lib_N+1023/1024),1024>>>(d_ArLib,Ar_State,Box,PI);
	cudaDeviceSynchronize();
	cudaMemcpy(ArLib,d_ArLib,sizeof(ArLib),cudaMemcpyDeviceToHost);
	cudaFree(d_ArLib);
  cout<<"*******Lib-Ar Initialization Done!*******\n";

}

/******************************************************************************/
__global__ void InitializationLib_Kernel(Atom *ArLib,Atom_State Ar_State,double *Box,double PI){
	double R1,R2;
	int tid=threadIdx.x+blockIdx.x*blockDim.x;

	if(tid<Lib_N){
		R1=random();
    ArLib[tid].x=Box[0][0]+(Box[0][1]-Box[0][0])*R1;
		R2=random();
    ArLib[tid].y=Box[1][0]+(Box[1][1]-Box[1][0])*R2;
    ArLib[tid].z=Box[2][1];
    R1=random();
    R2=random();
    ArLib[tid].vx=Ar_State.mp_V*sqrt(-log(R1))*cos(2*PI*R2);//Maxwell分布
		R1=random();
    R2=random();
    ArLib[tid].vy=Ar_State.mp_V*sqrt(-log(R1))*sin(2*PI*R2);
		R1=random();
    ArLib[tid].vz=-Ar_State.mp_V*sqrt(-log(R1));
		ArLib[tid].ax=0.0;
		ArLib[tid].ay=0.0;
		ArLib[tid].az=0.0;
	}
}

/******************************************************************************/
void Parameters::AssignSamples(){
	double *d_ArLib;
	int startid,endid,i,;
	__global__ void AssignSamples(Atom *ArLib,Atom_State Ar_State,double *Box,double PI);

	Block_N=SampleParallel_N;
	startid=0;
	endid=SampleParallel_N-1;
	i=0;
	cudaMalloc((void**)&d_Pt,sizeof(Pt));
	cudaMalloc((void**)&d_Pt_Init,sizeof(Pt));
	cudaMalloc((void**)&d_Ar,sizeof(Ar));
	while(startid<Lib_N){
		cudaMemcpy(d_Pt,Pt,sizeof(Pt),cudaMemcpyHostToDevice);
		cudaMemcpy(d_Pt_Init,Pt_Init,sizeof(Pt),cudaMemcpyHostToDevice);
		cudaMemcpy(d_Ar,ArLib[startid:endid],sizeof(Ar),cudaMemcpyHostToDevice);
		while(i<Tt){
			Verlet();
			Exit();
		}
		startid+=SampleParallel_N;
		endid+=SampleParallel_N;
		if(endid>=Lib_N){
			endid=Lib_N-1;
		}
	}
	cudaMemcpy(ArLib,d_ArLib,sizeof(ArLib),cudaMemcpyDeviceToHost);
	cudaFree(d_ArLib);
  cout<<"*******Lib-Ar Initialization Done!*******\n";

}

/******************************************************************************/
__global__ void SampleMD(){
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int bid=blockIdx.x;

  if(tid<Pt_State.N+Ar_State.N){
		if(tid<Pt_State.N){
			//Verlet_Pos()
			Pt[tid].x+=Pt[tid].vx*dt+0.5*Pt[tid].ax*dt**2;
			Pt[tid].y+=Pt[tid].vy*dt+0.5*Pt[tid].ay*dt**2;
			Pt[tid].z+=Pt[tid].vz*dt+0.5*Pt[tid].az*dt**2;
			//Pos_period()
			//X方向
			if(Pt[tid].x<Box[0][0]){
				Pt[tid].x+=Box[0][1]-Box[0][0];
				Pt_Init[tid].x+=Box[0][1]-Box[0][0];
			}
			else if(Pt[tid].x>=Box[0][1]){
				Pt[tid].x-=Box[0][1]-Box[0][0];
				Pt_Init[tid].x-=Box[0][1]-Box[0][0];
			}
			//Y方向
			if(Pt[tid].y<Box[1][0]){
				Pt[tid].y+=Box[1][1]-Box[1][0];
				Pt_Init[tid].y+=Box[1][1]-Box[1][0];
			}
			else if(Pt[tid].y>=Box[1][1]){
				Pt[tid].y-=Box[1][1]-Box[1][0];
				Pt_Init[tid].y-=Box[1][1]-Box[1][0];
			//Z方向下边界更新
			atomicMin(&Box[2][0],Pt[tid].z);
			}
		}
		else{
			//Verlet_Pos()
			Ar[bid].x+=Ar[bid].vx*dt+0.5*Ar[bid].ax*dt**2;
			Ar[bid].y+=Ar[bid].vy*dt+0.5*Ar[bid].ay*dt**2;
			Ar[bid].z+=Ar[bid].vz*dt+0.5*Ar[bid].az*dt**2;
			//Pos_period()
			//X方向
			if(Ar[bid].x<Box[0][0]){
				Ar[bid].x+=Box[0][1]-Box[0][0];
			}
			else if(Ar[bid].x>=Box[0][1]){
				Ar[bid].x-=Box[0][1]-Box[0][0];
			}
			//Y方向
			if(Ar[bid].y<Box[1][0]){
				Ar[bid].y+=Box[1][1]-Box[1][0];
			}
			else if(Ar[bid].y>=Box[1][1]){
				Ar[bid].y-=Box[1][1]-Box[1][0];
			}
		}
	}
}

/******************************************************************************/
void Parameters::Dump(double Pt_Pos[][3], double Ar_Pos[][3], int timestep, int ds) {
	int i;

	if (timestep%ds == 0) {
		ofstream MD;
		MD.open("Kernel_MD.dump", ios::app);
		MD << "ITEM: TIMESTEP\n";
		MD << timestep << "\n";
		MD << "ITEM: NUMBER OF ATOMS\n";
		MD << Pt_N + Ar_N << "\n";
		MD << "ITEM: BOX BOUNDS pp pp ff\n";
		MD << Box[0][0] << " " << Box[0][1] << "\n";
		MD << Box[1][0] << " " << Box[1][1] << "\n";
		MD << Box[2][0] << " " << Box[2][1] << "\n";
		MD << "ITEM: ATOMS id type x y z\n";
		for (i = 0;i < Pt_N;i++) {
			MD << i + 1 << " " << Pt_type[i] + 1 << " " << Pt_Pos[i][0] << " " << Pt_Pos[i][1] << " " << Pt_Pos[i][2] << "\n";
		}
		MD << Pt_N + 1 << " " << Ar_type[0] + 1 << " " << Ar_Pos[0][0] << " " << Ar_Pos[0][1] << " " << Ar_Pos[0][2] << "\n";
		MD.close();
		ofstream Zt;
		Zt.open("Kernel_MD_Zt.dat", ios::app);
		Zt << timestep * dt << " " << Ar_Pos[0][2] << "\n";
		Zt.close();
	}
}

/******************************************************************************/
void Parameters::Exit() {

	if (Ar.z[0][2] > d || timestep >= Tt) {
		state = false;
		Dump(Pt_Pos, Ar_Pos, timestep);
	}
	else {
		Dump(Pt_Pos, Ar_Pos, timestep, dumpstep);
	}
}

/******************************************************************************/
double random() {
	double R;
	R = 0.;
	while (R == 0.) {
		R = rand() / double(RAND_MAX);
	}
	return R;
}

////////////////////////////////////////////////////////////////////////////////
/*************************************main*************************************/
////////////////////////////////////////////////////////////////////////////////
int main() {
	class Parameters Pars;
	clock_t start, finish;
	double tperl;
	double Pt_Pos[4 * Pt_I*Pt_J*Pt_K][3], Pt_Vel[4 * Pt_I*Pt_J*Pt_K][3], Pt_Acc[4 * Pt_I*Pt_J*Pt_K][3], Ar_Pos[1][3], Ar_Vel[1][3], Ar_Acc[1][3];
	int timestep;

	Pars.Init();
	timestep = Pars.Initialization(Pt_Pos, Pt_Vel, Pt_Acc, Ar_Pos, Ar_Vel, Ar_Acc);
	Pars.Dump(Pt_Pos, Ar_Pos, timestep);
	start = clock();
	while (Pars.state) {
		Pars.Verlet(Ar_Pos, Ar_Vel, Ar_Acc, Pt_Pos, Pt_Vel, Pt_Acc);
		Pars.rescale_T(Pt_Vel);
		timestep += 1;
		Pars.Exit(Pt_Pos, Ar_Pos, timestep);
		finish = clock();
		tperl = double(finish - start) / CLOCKS_PER_SEC / timestep;
		cout << timestep << " TimeSteps; ArgTime: " << tperl << " Seconds!\r";
	}
	system("pause");
	return 0;
}
