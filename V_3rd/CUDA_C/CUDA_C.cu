#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <algorithm>
using namespace std;
//const int Lib_N = 100000;
//const int SampleParallel_N = 1000;
const int Atom_type_N = 2;
const int Pair_type_N = 3;
const int Pt_I = 6, Pt_J = 6, Pt_K = 3;
const int Pt_N = 4 * Pt_I * Pt_J * Pt_K;
const int Ar_N = 1;

class Parameters {
public:
	int Rt, Tt, dumpstep;
	double Mass[Atom_type_N], T[Atom_type_N], mp_V[Atom_type_N], LJ_E[Pair_type_N], LJ_S[Pair_type_N], Box_x[2], Box_y[2], Box_z[2], Pt_ePos_x[Pt_N], Pt_ePos_y[Pt_N], Pt_ePos_z[Pt_N], Pt_argVel[3];
	double PI, kB, fcc_lattice, nd_Mass, nd_Energy, nd_Length, nd_Velocity, nd_Time, nd_Acceleration, cutoff, d, spr_k, dt, Pt_T;
	bool state;
	void Init();
	void Initialization(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], double All_Vel_x[], double All_Vel_y[], double All_Vel_z[], double All_Acc_x[], double All_Acc_y[], double All_Acc_z[]);
	void Initialization_Kernel(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], double All_Vel_x[], double All_Vel_y[], double All_Vel_z[], double All_Acc_x[], double All_Acc_y[], double All_Acc_z[]);
	void rescale_T1(double All_Vel_x[], double All_Vel_y[], double All_Vel_z[]);
	void rescale_T3(double All_Vel_x[], double All_Vel_y[], double All_Vel_z[]);
	void Dump(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], int timestep, int ds = 1);
	void Exit(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], int timestep);
	double random();
};
void Parameters::Init() {
	//
	PI = 3.14159265;
	kB = 1.38E-23;
	Mass[0] = 39.95 / 6.02 * 1E-26;//kg
	Mass[1] = 195.08 / 6.02 * 1E-26;
	LJ_E[0] = 1.654E-21;//J
	LJ_E[1] = 5.207E-20;
	LJ_E[2] = 1.093E-21;
	LJ_S[0] = 3.40 * 1E-10;//m
	LJ_S[1] = 2.47 * 1E-10;
	LJ_S[2] = 2.94 * 1E-10;
	cutoff = 10 * 1E-10;
	fcc_lattice = 3.93E-10;
	T[0] = 300.;
	T[1] = 300.;
	mp_V[0] = sqrt(2 * kB*T[0] / Mass[0]);//
	mp_V[1] = sqrt(3 * kB*T[1] / Mass[1]);//
	//
	nd_Mass = Mass[1];
	nd_Energy = LJ_E[1];
	nd_Length = LJ_S[1];
	nd_Velocity = sqrt(nd_Energy / nd_Mass);
	nd_Time = nd_Length / nd_Velocity;
	nd_Acceleration = nd_Energy / (nd_Mass * nd_Length);
	//
	Mass[0] /= nd_Mass;
	Mass[1] /= nd_Mass;
	LJ_E[0] /= nd_Energy;
	LJ_E[1] /= nd_Energy;
	LJ_E[2] /= nd_Energy;
	LJ_S[0] /= nd_Length;
	LJ_S[1] /= nd_Length;
	LJ_S[2] /= nd_Length;
	cutoff /= nd_Length;
	fcc_lattice /= nd_Length;
	mp_V[0] /= nd_Velocity;
	mp_V[1] /= nd_Velocity;
	d = 5.0;
	spr_k = 5000.;
	dt = 0.001;
	Rt = 100;
	Tt = 35;
	dumpstep = 1;
	//
	Box_x[0] = 0;
	Box_x[1] = Pt_I * fcc_lattice;
	Box_y[0] = 0;
	Box_y[1] = Pt_J * fcc_lattice;
	Box_z[0] = -(Pt_K - 0.5)*fcc_lattice;
	Box_z[1] = d;
	//
	state = true;

	cout << "*******Parameters Initialized!*******\n";
}

/******************************************************************************/
void Parameters::Initialization(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], double All_Vel_x[], double All_Vel_y[], double All_Vel_z[], double All_Acc_x[], double All_Acc_y[], double All_Acc_z[]) {
	int i;
	double *d_Pt_argVel, *d_Pt_T, *d_Mass, *d_T, *d_LJ_E, *d_LJ_S, *d_cutoff, *d_spr_k;
	double *d_All_Pos_x, *d_All_Pos_y, *d_All_Pos_z, *d_All_Vel_x, *d_All_Vel_y, *d_All_Vel_z, *d_All_Acc_x, *d_All_Acc_y, *d_All_Acc_z, *d_Box_x, *d_Box_y, *d_Box_z, *d_Pt_ePos_x, *d_Pt_ePos_y, *d_Pt_ePos_z;
	__global__ void Pos_period(double *All_Pos_x, double *All_Pos_y, double *Box_x, double *Box_y, double *Pt_ePos_x, double *Pt_ePos_y);
	__global__ void rescale_T2(double *All_Vel_x, double *All_Vel_y, double *All_Vel_z, double *Pt_argVel);
	__global__ void rescale_T4(double *All_Vel_x, double *All_Vel_y, double *All_Vel_z, double *T, double *Pt_T);
	__global__ void Acceleration_period(double *All_Pos_x, double *All_Pos_y, double *All_Pos_z, double *All_Acc_x, double *All_Acc_y, double *All_Acc_z, double *LJ_E, double *LJ_S, double *Box_x, double *Box_y, double *Box_z, double *cutoff, double *Pt_ePos_x, double *Pt_ePos_y, double *Pt_ePos_z, double *spr_k, double *Mass);

	cout << "Box_ZoneX: " << Box_x[0] << ", " << Box_x[1] << "\n";
	cout << "Box_ZoneY: " << Box_y[0] << ", " << Box_y[1] << "\n";
	cout << "Box_ZoneZ: " << Box_z[0] << ", " << Box_z[1] << "\n";
	//
	Initialization_Kernel(All_type, All_Pos_x, All_Pos_y, All_Pos_z, All_Vel_x, All_Vel_y, All_Vel_z, All_Acc_x, All_Acc_y, All_Acc_z);
	//
	cudaMalloc((void**)&d_All_Pos_x, sizeof(All_Pos_x));
	cudaMalloc((void**)&d_All_Pos_y, sizeof(All_Pos_y));
	cudaMalloc((void**)&d_All_Pos_z, sizeof(All_Pos_z));
	cudaMalloc((void**)&d_All_Vel_x, sizeof(All_Vel_x));
	cudaMalloc((void**)&d_All_Vel_y, sizeof(All_Vel_y));
	cudaMalloc((void**)&d_All_Vel_z, sizeof(All_Vel_z));
	cudaMalloc((void**)&d_All_Acc_x, sizeof(All_Acc_x));
	cudaMalloc((void**)&d_All_Acc_y, sizeof(All_Acc_y));
	cudaMalloc((void**)&d_All_Acc_z, sizeof(All_Acc_z));
	cudaMalloc((void**)&d_Box_x, sizeof(Box_x));
	cudaMalloc((void**)&d_Box_y, sizeof(Box_y));
	cudaMalloc((void**)&d_Box_z, sizeof(Box_z));
	cudaMalloc((void**)&d_Pt_ePos_x, sizeof(Pt_ePos_x));
	cudaMalloc((void**)&d_Pt_ePos_y, sizeof(Pt_ePos_y));
	cudaMalloc((void**)&d_Pt_ePos_z, sizeof(Pt_ePos_z));
	cudaMalloc((void**)&d_Mass, sizeof(Mass));
	cudaMalloc((void**)&d_T, sizeof(T));
	cudaMalloc((void**)&d_LJ_E, sizeof(LJ_E));
	cudaMalloc((void**)&d_LJ_S, sizeof(LJ_S));
	cudaMalloc((void**)&d_cutoff, sizeof(double));
	cudaMalloc((void**)&d_spr_k, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVel, sizeof(Pt_argVel));
	cudaMalloc((void**)&d_Pt_T, sizeof(double));
	//
	cudaMemcpy(d_All_Pos_x, All_Pos_x, sizeof(All_Pos_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Pos_y, All_Pos_y, sizeof(All_Pos_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Pos_z, All_Pos_z, sizeof(All_Pos_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel_x, All_Vel_x, sizeof(All_Vel_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel_y, All_Vel_y, sizeof(All_Vel_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel_z, All_Vel_z, sizeof(All_Vel_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc_x, All_Acc_x, sizeof(All_Acc_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc_y, All_Acc_y, sizeof(All_Acc_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc_z, All_Acc_z, sizeof(All_Acc_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box_x, Box_x, sizeof(Box_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box_y, Box_y, sizeof(Box_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box_z, Box_z, sizeof(Box_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_ePos_x, Pt_ePos_x, sizeof(Pt_ePos_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_ePos_y, Pt_ePos_y, sizeof(Pt_ePos_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_ePos_z, Pt_ePos_z, sizeof(Pt_ePos_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mass, Mass, sizeof(Mass), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, T, sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_E, LJ_E, sizeof(LJ_E), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_S, LJ_S, sizeof(LJ_S), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cutoff, &cutoff, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_spr_k, &spr_k, sizeof(double), cudaMemcpyHostToDevice);
	//
	Pos_period << <1, Pt_N + Ar_N >> >(d_All_Pos_x, d_All_Pos_y, d_Box_x, d_Box_y, d_Pt_ePos_x, d_Pt_ePos_y);
	cudaDeviceSynchronize();
	//
	Box_z[0]=All_Pos_z[0];
	for(i=0;i<Pt_N;i++){
		if(All_Pos_z[i]<Box_z[0]){
			Box_z[0]=All_Pos_z[i];
		}
	}
	cout<<Box_z[0]<<"\n";
	cudaMemcpy(d_Box_z, Box_z, sizeof(Box_z), cudaMemcpyHostToDevice);
	//
	rescale_T1(All_Vel_x, All_Vel_y, All_Vel_z);
	cudaMemcpy(d_Pt_argVel, Pt_argVel, sizeof(Pt_argVel), cudaMemcpyHostToDevice);
	rescale_T2 << <1, Pt_N >> > (d_All_Vel_x, d_All_Vel_y, d_All_Vel_z, d_Pt_argVel);
	cudaDeviceSynchronize();
	cudaMemcpy(All_Vel_x, d_All_Vel_x, sizeof(All_Vel_x), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Vel_y, d_All_Vel_y, sizeof(All_Vel_y), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Vel_z, d_All_Vel_z, sizeof(All_Vel_z), cudaMemcpyDeviceToHost);
	rescale_T3(All_Vel_x, All_Vel_y, All_Vel_z);
	cudaMemcpy(d_Pt_T, &Pt_T, sizeof(double), cudaMemcpyHostToDevice);
	rescale_T4 << <1, Pt_N >> > (d_All_Vel_x, d_All_Vel_y, d_All_Vel_z, d_T, d_Pt_T);
	cudaDeviceSynchronize();
	//
	Acceleration_period << <1, Pt_N + Ar_N >> >(d_All_Pos_x, d_All_Pos_y, d_All_Pos_z, d_All_Acc_x, d_All_Acc_y, d_All_Acc_z, d_LJ_E, d_LJ_S, d_Box_x, d_Box_y, d_Box_z, d_cutoff, d_Pt_ePos_x, d_Pt_ePos_y, d_Pt_ePos_z, d_spr_k, d_Mass);
	cudaDeviceSynchronize();
	cudaMemcpy(All_Pos_x, d_All_Pos_x, sizeof(All_Pos_x), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Pos_y, d_All_Pos_y, sizeof(All_Pos_y), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Pos_z, d_All_Pos_z, sizeof(All_Pos_z), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Vel_x, d_All_Vel_x, sizeof(All_Vel_x), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Vel_y, d_All_Vel_y, sizeof(All_Vel_y), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Vel_z, d_All_Vel_z, sizeof(All_Vel_z), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Acc_x, d_All_Acc_x, sizeof(All_Acc_x), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Acc_y, d_All_Acc_y, sizeof(All_Acc_y), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Acc_z, d_All_Acc_z, sizeof(All_Acc_z), cudaMemcpyDeviceToHost);
	cudaMemcpy(Pt_ePos_x, d_Pt_ePos_x, sizeof(Pt_ePos_x), cudaMemcpyDeviceToHost);
	cudaMemcpy(Pt_ePos_y, d_Pt_ePos_y, sizeof(Pt_ePos_y), cudaMemcpyDeviceToHost);
	cudaMemcpy(Pt_ePos_z, d_Pt_ePos_z, sizeof(Pt_ePos_z), cudaMemcpyDeviceToHost);
	//
	cout << "Created " << Pt_N << " Pt\n";
	cout << "Created " << Ar_N << " Ar\n";
	cout << "Pt Average Speed in X: " << Pt_argVelx << "\n";
	cout << "Pt Average Speed in Y: " << Pt_argVely << "\n";
	cout << "Pt Average Speed in Z: " << Pt_argVelz << "\n";
	cout << "Pt Temperature: " << Pt_T << "\n";
	cout << "Ar Incidence Speed: " << All_Vel_x[Pt_N] << "," << All_Vel_y[Pt_N] << "," << All_Vel_z[Pt_N] << "\n";
	cout << "*******Model Initialization Done!*******\n";
	cudaFree(d_All_Pos_x);
	cudaFree(d_All_Pos_y);
	cudaFree(d_All_Pos_z);
	cudaFree(d_All_Vel_x);
	cudaFree(d_All_Vel_y);
	cudaFree(d_All_Vel_z);
	cudaFree(d_All_Acc_x);
	cudaFree(d_All_Acc_y);
	cudaFree(d_All_Acc_z);
	cudaFree(d_Box_x);
	cudaFree(d_Box_y);
	cudaFree(d_Box_z);
	cudaFree(d_Pt_ePos_x);
	cudaFree(d_Pt_ePos_y);
	cudaFree(d_Pt_ePos_z);
	cudaFree(d_nd_Velocity);
	cudaFree(d_Mass);
	cudaFree(d_nd_Mass);
	cudaFree(d_kB);
	cudaFree(d_T);
	cudaFree(d_LJ_E);
	cudaFree(d_LJ_S);
	cudaFree(d_cutoff);
	cudaFree(d_spr_k);
	cudaFree(d_Pt_argVel);
	cudaFree(d_Pt_T);
}

/******************************************************************************/
void Parameters::Initialization_Kernel(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], double All_Vel_x[], double All_Vel_y[], double All_Vel_z[], double All_Acc_x[], double All_Acc_y[], double All_Acc_z[]) {
	int i, j, k, axis, count;
	double R1, R2, Rx, Ry;

	count = 0;
	srand((unsigned)time(NULL));
	for (i = 0; i < 2 * Pt_I; i++) {
		for (j = 0; j < 2 * Pt_J; j++) {
			for (k = 0; k < 2 * Pt_K; k++) {
				if (i / 2. + j / 2. + k / 2. == int(i / 2. + j / 2. + k / 2.)) {
					All_type[count] = 1;
					All_Pos_x[count] = i / 2.*fcc_lattice;
					Pt_ePos_x[count] = All_Pos_x[count];
					All_Pos_y[count] = j / 2.*fcc_lattice;
					Pt_ePos_y[count] = All_Pos_y[count];
					All_Pos_z[count] = (k / 2. - 2.5)*fcc_lattice;
					Pt_ePos_z[count] = All_Pos_z[count];
					R1 = random();
					R2 = random();
					All_Vel_x[count] = mp_V[1] / sqrt(3)*sqrt(-2 * log(R1))*cos(2 * PI*R2);
					R1 = random();
					R2 = random();
					All_Vel_y[count] = mp_V[1] / sqrt(3)*sqrt(-2 * log(R1))*cos(2 * PI*R2);
					R1 = random();
					R2 = random();
					All_Vel_z[count] = mp_V[1] / sqrt(3)*sqrt(-2 * log(R1))*cos(2 * PI*R2);
					All_Acc_x[count] = 0.0;
					All_Acc_y[count] = 0.0;
					All_Acc_z[count] = 0.0;
					count += 1;
				}
			}
		}
	}
	Rx = random();
	Ry = random();
	All_type[count] = 0;
	All_Pos_x[count] = Box_x[0] + (Box_x[1] - Box_x[0]) * Rx;
	All_Pos_y[count] = Box_y[0] + (Box_y[1] - Box_y[0]) * Ry;
	All_Pos_z[count] = Box_z[1];
	R1 = random();
	R2 = random();
	All_Vel_x[count] = mp_V[0] * sqrt(-log(R1))*cos(2 * PI*R2);//Maxwell�ֲ�
	R1 = random();
	R2 = random();
	All_Vel_y[count] = mp_V[0] * sqrt(-log(R1))*sin(2 * PI*R2);
	R1 = random();
	All_Vel_z[count] = -mp_V[0] * sqrt(-log(R1));
	All_Acc_x[count] = 0.0;
	All_Acc_y[count] = 0.0;
	All_Acc_z[count] = 0.0;
}

/******************************************************************************/
__global__ void Pos_period(double *All_Pos_x, double *All_Pos_y, double *Box_x, double *Box_y, double *Pt_ePos_x, double *Pt_ePos_y) {
	int tid = threadIdx.x;

	if (tid<Pt_N + Ar_N) {
		//X
		if (All_Pos_x[tid]<Box_x[0]) {
			All_Pos_x[tid] += Box_x[1] - Box_x[0];
			if (tid<Pt_N) {
				Pt_ePos_x[tid] += Box_x[1] - Box_x[0];
			}
		}
		else if (All_Pos_x[tid] >= Box_x[1]) {
			All_Pos_x[tid] -= Box_x[1] - Box_x[0];
			if (tid<Pt_N) {
				Pt_ePos_x[tid] -= Box_x[1] - Box_x[0];
			}
		}
		//Y
		if (All_Pos_y[tid]<Box_y[0]) {
			All_Pos_y[tid] += Box_y[1] - Box_y[0];
			if (tid<Pt_N) {
				Pt_ePos_y[tid] += Box_y[1] - Box_y[0];
			}
		}
		else if (All_Pos_y[tid] >= Box_y[1]) {
			All_Pos_y[tid] -= Box_y[1] - Box_y[0];
			if (tid<Pt_N) {
				Pt_ePos_y[tid] -= Box_y[1] - Box_y[0];
			}
		}
	}
}

/******************************************************************************/
void Parameters::rescale_T1(double All_Vel_x[], double All_Vel_y[], double All_Vel_z[]) {
	int i;

	Pt_argVel[0] = 0.0;
	Pt_argVel[1] = 0.0;
	Pt_argVel[2] = 0.0;
	for (i = 0; i < Pt_N; i++) {
		Pt_argVel[0] += All_Vel_x[i] / Pt_N;
		Pt_argVel[1] += All_Vel_y[i] / Pt_N;
		Pt_argVel[2] += All_Vel_z[i] / Pt_N;
	}
}

/******************************************************************************/
__global__ void rescale_T2(double *All_Vel_x, double *All_Vel_y, double *All_Vel_z, double *Pt_argVel) {
	int tid = threadIdx.x;

	if (tid<Pt_N) {
		All_Vel_x[tid] -= Pt_argVel[0];
		All_Vel_y[tid] -= Pt_argVel[1];
		All_Vel_z[tid] -= Pt_argVel[2];
	}
}

/******************************************************************************/
void Parameters::rescale_T3(double All_Vel_x[], double All_Vel_y[], double All_Vel_z[]) {
	int i;

	Pt_T = 0.0;
	for (i = 0; i < Pt_N; i++) {
		Pt_T += All_Vel_x[i] * All_Vel_x[i] + All_Vel_y[i] * All_Vel_y[i] + All_Vel_z[i] * All_Vel_z[i];
	}
	Pt_T *= nd_Velocity * nd_Velocity * Mass[1] * nd_Mass / (3 * Pt_N * kB);
}

/******************************************************************************/
__global__ void rescale_T4(double *All_Vel_x, double *All_Vel_y, double *All_Vel_z, double *T, double *Pt_T) {
	int tid = threadIdx.x;

	if (tid<Pt_N) {
		All_Vel_x[tid] *= sqrt(T[1] / (*Pt_T));
		All_Vel_y[tid] *= sqrt(T[1] / (*Pt_T));
		All_Vel_z[tid] *= sqrt(T[1] / (*Pt_T));
	}
}

/******************************************************************************/
__global__ void Acceleration_period(double *All_Pos_x, double *All_Pos_y, double *All_Pos_z, double *All_Acc_x, double *All_Acc_y, double *All_Acc_z, double *LJ_E, double *LJ_S, double *Box_x, double *Box_y, double *Box_z, double *cutoff, double *Pt_ePos_x, double *Pt_ePos_y, double *Pt_ePos_z, double *spr_k, double *Mass) {
	int i, LJ_pair;
	double Epair, Spair, Pairx, Pairy, Pairz, Dispair, Fpair, Atom_Fx, Atom_Fy, Atom_Fz;
	double Spring_Disx, Spring_Fx, Pt_Fx, Spring_Disy, Spring_Fy, Pt_Fy, Spring_Disz, Spring_Fz, Pt_Fz, Ar_Fx, Ar_Fy, Ar_Fz;
	int tid = threadIdx.x;

	if (tid<Pt_N + Ar_N) {
		Atom_Fx = 0.0;
		Atom_Fy = 0.0;
		Atom_Fz = 0.0;
		for (i = 0; i<Pt_N + Ar_N; i++) {
			if (tid<Pt_N && i<Pt_N) {
				LJ_pair = 1;
			}
			else if (tid >= Pt_N && i >= Pt_N) {
				LJ_pair = 0;
			}
			else {
				LJ_pair = 2;
			}
			Epair = LJ_E[LJ_pair];
			Spair = LJ_S[LJ_pair];
			//
			Pairx = All_Pos_x[tid] - All_Pos_x[i];
			Pairy = All_Pos_y[tid] - All_Pos_y[i];
			Pairz = All_Pos_z[tid] - All_Pos_z[i];
			if (abs(Pairx) >= Box_x[1] - Box_x[0] - (*cutoff)) {
				Pairx -= (Box_x[1] - Box_x[0])*Pairx / abs(Pairx);
			}
			if (abs(Pairy) >= Box_y[1] - Box_y[0] - (*cutoff)) {
				Pairy -= (Box_y[1] - Box_y[0])*Pairy / abs(Pairy);
			}
			//
			Dispair = sqrt(Pairx * Pairx + Pairy * Pairy + Pairz * Pairz);
			if (Dispair > 0 && Dispair <= (*cutoff)) {
				Fpair = 48 * Epair*(pow(Spair, 12) / pow(Dispair, 13) - 0.5*pow(Spair, 6) / pow(Dispair, 7));
				Atom_Fx += Pairx * Fpair / Dispair;
				Atom_Fy += Pairy * Fpair / Dispair;
				Atom_Fz += Pairz * Fpair / Dispair;
			}
		}
		if (tid<Pt_N) {
			//Pt
			Spring_Disx = All_Pos_x[tid] - Pt_ePos_x[tid];
			Spring_Fx = -(*spr_k) * Spring_Disx;
			Pt_Fx = Atom_Fx + Spring_Fx;
			All_Acc_x[tid] = Pt_Fx / Mass[1];
			Spring_Disy = All_Pos_y[tid] - Pt_ePos_y[tid];
			Spring_Fy = -(*spr_k) * Spring_Disy;
			Pt_Fy = Atom_Fy + Spring_Fy;
			All_Acc_y[tid] = Pt_Fy / Mass[1];
			Spring_Disz = All_Pos_z[tid] - Pt_ePos_z[tid];
			Spring_Fz = -(*spr_k) * Spring_Disz;
			Pt_Fz = Atom_Fz + Spring_Fz;
			All_Acc_z[tid] = Pt_Fz / Mass[1];
		}
		else {
			//Ar
			Ar_Fx = Atom_Fx;
			All_Acc_x[tid] = Ar_Fx / Mass[0];
			Ar_Fy = Atom_Fy;
			All_Acc_y[tid] = Ar_Fy / Mass[0];
			Ar_Fz = Atom_Fz;
			All_Acc_z[tid] = Ar_Fz / Mass[0];
		}
	}
}

/******************************************************************************/
__global__ void Verlet_Pos(double *All_Pos_x, double *All_Pos_y, double *All_Pos_z, double *All_Vel_x, double *All_Vel_y, double *All_Vel_z, double *All_Acc_x, double *All_Acc_y, double *All_Acc_z, double *dt) {
	int tid = threadIdx.x;

	if (tid<Pt_N + Ar_N) {
		All_Pos_x[tid] += All_Vel_x[tid] * (*dt) + 0.5*All_Acc_x[tid] * (*dt) * (*dt);
		All_Pos_y[tid] += All_Vel_y[tid] * (*dt) + 0.5*All_Acc_y[tid] * (*dt) * (*dt);
		All_Pos_z[tid] += All_Vel_z[tid] * (*dt) + 0.5*All_Acc_z[tid] * (*dt) * (*dt);
	}
}

/******************************************************************************/
__global__ void Verlet_Vel(double *All_Vel_x, double *All_Vel_y, double *All_Vel_z, double *All_Acc_temp_x, double *All_Acc_temp_y, double *All_Acc_temp_z, double *All_Acc_x, double *All_Acc_y, double *All_Acc_z, double *dt) {
	int tid = threadIdx.x;

	if (tid<Pt_N + Ar_N) {
		All_Vel_x[tid] += 0.5*(All_Acc_temp_x[tid] + All_Acc_x[tid])*(*dt);
		All_Vel_y[tid] += 0.5*(All_Acc_temp_y[tid] + All_Acc_y[tid])*(*dt);
		All_Vel_z[tid] += 0.5*(All_Acc_temp_z[tid] + All_Acc_z[tid])*(*dt);
	}
}

/******************************************************************************/
void Parameters::Dump(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], int timestep, int ds) {
	int i;

	if (timestep%ds == 0) {
		ofstream MD;
		MD.open("Kernel_MD_CUDA_C.dump", ios::app);
		MD << "ITEM: TIMESTEP\n";
		MD << timestep << "\n";
		MD << "ITEM: NUMBER OF ATOMS\n";
		MD << Pt_N + Ar_N << "\n";
		MD << "ITEM: BOX BOUNDS pp pp ff\n";
		MD << Box_x[0] << " " << Box_x[1] << "\n";
		MD << Box_y[0] << " " << Box_y[1] << "\n";
		MD << Box_z[0] << " " << Box_z[1] << "\n";
		MD << "ITEM: ATOMS id type x y z\n";
		for (i = 0; i < Pt_N + Ar_N; i++) {
			MD << i + 1 << " " << All_type[i] + 1 << " " << All_Pos_x[i] << " " << All_Pos_y[i] << " " << All_Pos_z[i] << "\n";
		}
		MD.close();
		ofstream Zt;
		Zt.open("Kernel_MD_CUDA_C_Zt.dat", ios::app);
		Zt << timestep * dt << " " << All_Pos_z[Pt_N] << "\n";
		Zt.close();
	}
}

/******************************************************************************/
void Parameters::Exit(int All_type[], double All_Pos_x[], double All_Pos_y[], double All_Pos_z[], int timestep) {

	if (All_Pos_z[Pt_N] > d || timestep >= Tt) {
		state = false;
		Dump(All_type, All_Pos_x, All_Pos_y, All_Pos_z, timestep);
	}
	else {
		Dump(All_type, All_Pos_x, All_Pos_y, All_Pos_z, timestep, dumpstep);
	}
}

/******************************************************************************/
double Parameters::random() {
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
	int All_type[Pt_N + Ar_N];
	double All_Pos_x[Pt_N + Ar_N], All_Pos_y[Pt_N + Ar_N], All_Pos_z[Pt_N + Ar_N], All_Vel_x[Pt_N + Ar_N], All_Vel_y[Pt_N + Ar_N], All_Vel_z[Pt_N + Ar_N], All_Acc_x[Pt_N + Ar_N], All_Acc_y[Pt_N + Ar_N], All_Acc_z[Pt_N + Ar_N];
	double *d_All_Pos_x, *d_All_Pos_y, *d_All_Pos_z, *d_All_Vel_x, *d_All_Vel_y, *d_All_Vel_z, *d_All_Acc_x, *d_All_Acc_y, *d_All_Acc_z, *d_All_Acc_temp_x, *d_All_Acc_temp_y, *d_All_Acc_temp_z, *d_Box_x, *d_Box_y, *d_Box_z, *d_Pt_ePos_x, *d_Pt_ePos_y, *d_Pt_ePos_z;
	double *d_dt, *d_Mass, *d_T, *d_LJ_E, *d_LJ_S, *d_cutoff, *d_spr_k, *d_Pt_argVel, *d_Pt_T;
	int timestep=0;

	Pars.Init();
	Pars.Initialization(All_type, All_Pos_x, All_Pos_y, All_Pos_z, All_Vel_x, All_Vel_y, All_Vel_z, All_Acc_x, All_Acc_y, All_Acc_z);
	Pars.Exit(All_type, All_Pos_x, All_Pos_y, All_Pos_z, timestep);
	//
	cudaMalloc((void**)&d_All_Pos_x, sizeof(All_Pos_x));
	cudaMalloc((void**)&d_All_Pos_y, sizeof(All_Pos_y));
	cudaMalloc((void**)&d_All_Pos_z, sizeof(All_Pos_z));
	cudaMalloc((void**)&d_All_Vel_x, sizeof(All_Vel_x));
	cudaMalloc((void**)&d_All_Vel_y, sizeof(All_Vel_y));
	cudaMalloc((void**)&d_All_Vel_z, sizeof(All_Vel_z));
	cudaMalloc((void**)&d_All_Acc_x, sizeof(All_Acc_x));
	cudaMalloc((void**)&d_All_Acc_y, sizeof(All_Acc_y));
	cudaMalloc((void**)&d_All_Acc_z, sizeof(All_Acc_z));
	cudaMalloc((void**)&d_All_Acc_temp_x, sizeof(All_Acc_x));
	cudaMalloc((void**)&d_All_Acc_temp_y, sizeof(All_Acc_y));
	cudaMalloc((void**)&d_All_Acc_temp_z, sizeof(All_Acc_z));
	cudaMalloc((void**)&d_Box_x, sizeof(Pars.Box_x));
	cudaMalloc((void**)&d_Box_y, sizeof(Pars.Box_y));
	cudaMalloc((void**)&d_Box_z, sizeof(Pars.Box_z));
	cudaMalloc((void**)&d_Pt_ePos_x, sizeof(Pars.Pt_ePos_x));
	cudaMalloc((void**)&d_Pt_ePos_y, sizeof(Pars.Pt_ePos_y));
	cudaMalloc((void**)&d_Pt_ePos_z, sizeof(Pars.Pt_ePos_z));
	cudaMalloc((void**)&d_dt, sizeof(double));
	cudaMalloc((void**)&d_Mass, sizeof(Pars.Mass));
	cudaMalloc((void**)&d_T, sizeof(Pars.T));
	cudaMalloc((void**)&d_LJ_E, sizeof(Pars.LJ_E));
	cudaMalloc((void**)&d_LJ_S, sizeof(Pars.LJ_S));
	cudaMalloc((void**)&d_cutoff, sizeof(double));
	cudaMalloc((void**)&d_spr_k, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVel, sizeof(double));
	cudaMalloc((void**)&d_Pt_T, sizeof(double));
	//
	cudaMemcpy(d_All_Pos_x, All_Pos_x, sizeof(All_Pos_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Pos_y, All_Pos_y, sizeof(All_Pos_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Pos_z, All_Pos_z, sizeof(All_Pos_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel_x, All_Vel_x, sizeof(All_Vel_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel_y, All_Vel_y, sizeof(All_Vel_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel_z, All_Vel_z, sizeof(All_Vel_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc_x, All_Acc_x, sizeof(All_Acc_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc_y, All_Acc_y, sizeof(All_Acc_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc_z, All_Acc_z, sizeof(All_Acc_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box_x, Pars.Box_x, sizeof(Pars.Box_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box_y, Pars.Box_y, sizeof(Pars.Box_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box_z, Pars.Box_z, sizeof(Pars.Box_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_ePos_x, Pars.Pt_ePos_x, sizeof(Pars.Pt_ePos_x), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_ePos_y, Pars.Pt_ePos_y, sizeof(Pars.Pt_ePos_y), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_ePos_z, Pars.Pt_ePos_z, sizeof(Pars.Pt_ePos_z), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dt, &Pars.dt, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mass, Pars.Mass, sizeof(Pars.Mass), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, Pars.T, sizeof(Pars.T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_E, Pars.LJ_E, sizeof(Pars.LJ_E), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_S, Pars.LJ_S, sizeof(Pars.LJ_S), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cutoff, &Pars.cutoff, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_spr_k, &Pars.spr_k, sizeof(double), cudaMemcpyHostToDevice);
	start = clock();
	while (Pars.state) {
		//
		Verlet_Pos << <1, Pt_N + Ar_N >> >(d_All_Pos_x, d_All_Pos_y, d_All_Pos_z, d_All_Vel_x, d_All_Vel_y, d_All_Vel_z, d_All_Acc_x, d_All_Acc_y, d_All_Acc_z, d_dt);
		cudaDeviceSynchronize();
		//
		Pos_period << <1, Pt_N + Ar_N >> >(d_All_Pos_x, d_All_Pos_y, d_Box_x, d_Box_y, d_Pt_ePos_x, d_Pt_ePos_y);
		cudaDeviceSynchronize();
		//
		d_All_Acc_temp_x = d_All_Acc_x;
		d_All_Acc_temp_y = d_All_Acc_y;
		d_All_Acc_temp_z = d_All_Acc_z;
		//
		Acceleration_period << <1, Pt_N + Ar_N >> >(d_All_Pos_x, d_All_Pos_y, d_All_Pos_z, d_All_Acc_x, d_All_Acc_y, d_All_Acc_z, d_LJ_E, d_LJ_S, d_Box_x, d_Box_y, d_Box_z, d_cutoff, d_Pt_ePos_x, d_Pt_ePos_y, d_Pt_ePos_z, d_spr_k, d_Mass);
		cudaDeviceSynchronize();
		//
		Verlet_Vel << <1, Pt_N + Ar_N >> >(d_All_Vel_x, d_All_Vel_y, d_All_Vel_z, d_All_Acc_temp_x, d_All_Acc_temp_y, d_All_Acc_temp_z, d_All_Acc_x, d_All_Acc_y, d_All_Acc_z, d_dt);
		cudaDeviceSynchronize();
		//
		cudaMemcpy(All_Vel_x, d_All_Vel_x, sizeof(All_Vel_x), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Vel_y, d_All_Vel_y, sizeof(All_Vel_y), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Vel_z, d_All_Vel_z, sizeof(All_Vel_z), cudaMemcpyDeviceToHost);
		Pars.rescale_T1(All_Vel_x, All_Vel_y, All_Vel_z);
		cudaMemcpy(d_Pt_argVel, Pars.Pt_argVel, sizeof(Pars.Pt_argVel), cudaMemcpyHostToDevice);
		rescale_T2 << <1, Pt_N >> > (d_All_Vel_x, d_All_Vel_y, d_All_Vel_z, d_Pt_argVel);
		cudaDeviceSynchronize();
		cudaMemcpy(All_Vel_x, d_All_Vel_x, sizeof(All_Vel_x), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Vel_y, d_All_Vel_y, sizeof(All_Vel_y), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Vel_z, d_All_Vel_z, sizeof(All_Vel_z), cudaMemcpyDeviceToHost);
		Pars.rescale_T3(All_Vel_x, All_Vel_y, All_Vel_z);
		cudaMemcpy(d_Pt_T, &Pars.Pt_T, sizeof(double), cudaMemcpyHostToDevice);
		rescale_T4 << <1, Pt_N >> > (d_All_Vel_x, d_All_Vel_y, d_All_Vel_z, d_T, d_Pt_T);
		cudaDeviceSynchronize();
		//
		cudaMemcpy(All_Pos_x, d_All_Pos_x, sizeof(All_Pos_x), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Pos_y, d_All_Pos_y, sizeof(All_Pos_y), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Pos_z, d_All_Pos_z, sizeof(All_Pos_z), cudaMemcpyDeviceToHost);
		//
		Box_z[0]=All_Pos_z[0];
		for(i=0;i<Pt_N;i++){
			if(All_Pos_z[i]<Box_z[0]){
				Box_z[0]=All_Pos_z[i];
			}
		}
		//
		timestep += 1;
		Pars.Exit(All_type, All_Pos_x, All_Pos_y, All_Pos_z, timestep);
		finish = clock();
		tperl = double(finish - start) / CLOCKS_PER_SEC / timestep;
		cout << timestep << " TimeSteps; ArgTime: " << tperl << " Seconds!\r";
	}
	cudaFree(d_All_Pos_x);
	cudaFree(d_All_Pos_y);
	cudaFree(d_All_Pos_z);
	cudaFree(d_All_Vel_x);
	cudaFree(d_All_Vel_y);
	cudaFree(d_All_Vel_z);
	cudaFree(d_All_Acc_x);
	cudaFree(d_All_Acc_y);
	cudaFree(d_All_Acc_z);
	cudaFree(d_All_Acc_temp_x);
	cudaFree(d_All_Acc_temp_y);
	cudaFree(d_All_Acc_temp_z);
	cudaFree(d_Box_x);
	cudaFree(d_Box_y);
	cudaFree(d_Box_z);
	cudaFree(d_Pt_ePos_x);
	cudaFree(d_Pt_ePos_y);
	cudaFree(d_Pt_ePos_z);
	cudaFree(d_dt);
	cudaFree(d_Mass);
	cudaFree(d_T);
	cudaFree(d_LJ_E);
	cudaFree(d_LJ_S);
	cudaFree(d_cutoff);
	cudaFree(d_spr_k);
	cudaFree(d_Pt_argVel);
	cudaFree(d_Pt_T);
	cout << "\n";
	system("pause");
	return 0;
}
