#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <algorithm>
using namespace std;
//Constants
const float PI = 3.14159265;
const float kB = 1.38E-23;
const int Atom_type_N = 2;
const int Pair_type_N = 3;
const int Pt_I = 6, Pt_J = 6, Pt_K = 3;
const int Pt_N = 4 * Pt_I * Pt_J * Pt_K;
const int Ar_N = 1;
//Customize Structures
struct Wall_Molecule {
	int Sid;
	int Lid;
	float x;
	float y;
	float z;
	float Bx;
	float By;
	float Bz;
	float vx;
	float vy;
	float vz;
	float ax;
	float ay;
	float az;
	float atx;
	float aty;
	float atz;
	float fx;
	float fy;
	float fz;
};
struct Gas_Molecule {
	int Sid;
	int Lid;
	float x;
	float y;
	float z;
	float vx;
	float vy;
	float vz;
	float ax;
	float ay;
	float az;
	float atx;
	float aty;
	float atz;
	float fx;
	float fy;
	float fz;
};
struct Parameters_Wall {
	int Type;
	float Mass;
	float T;
	float mpVelocity;
	float CurrentT;
	float ArgVelX;
	float ArgVelY;
	float ArgVelZ;
	float Sigma;
	float Epselon;
	float Lattice;
};
struct Parameters_Gas {
	int Type;
	float Mass;
	float T;
	float mpVelocity;
	float Sigma;
	float Epselon;
};
struct Parameters_MD {
	float ArPt_Sigma;
	float ArPt_Epselon;
	float CutOff;
	float SprK;
	float dt;
	int Rt;
	int Tt;
	int DumpStep;
	int TimeStep;
	float BoxXLow;
	float BoxXHigh;
	float BoxYLow;
	float BoxYHigh;
	float BoxZLow;
	float BoxZHigh;
	bool State;
};
struct Dimensionless {
	float Mass;
	float Energy;
	float Length;
	float Velocity;
	float Time;
	float Acceleration;
};
//MD Definition
class MD {
	Parameters_Wall Pars_Pt;
	Parameters_Gas Pars_Ar;
	Parameters_MD Pars_MD;
	Dimensionless Pars_Dim;
	Wall_Molecule Pt[Pt_N];
	Gas_Molecule Ar[Ar_N];
public:
	void Pars_Init();
	void Models_Init();
	void Init_Kernels();
	void RescaleT1();
	void RescaleT3();
	void MainMD();
	void Dump();
	void Exit();
	float random();
};

/******************************************************************************/
void MD::Pars_Init() {
	//
	Pars_Ar.Type = 1;
	Pars_Pt.Type = 2;
	Pars_Ar.Mass = 39.95 / 6.02 * 1E-26;//kg
	Pars_Pt.Mass = 195.08 / 6.02 * 1E-26;
	Pars_Ar.T = 300.0;//K
	Pars_Pt.T = 300.0;
	Pars_Ar.mpVelocity = sqrt(2 * kB * Pars_Ar.T / Pars_Ar.Mass);//
	Pars_Pt.mpVelocity = sqrt(3 * kB * Pars_Pt.T / Pars_Pt.Mass);//
	Pars_Pt.CurrentT = 0.0;
	Pars_Pt.ArgVelX = 0.0;
	Pars_Pt.ArgVelY = 0.0;
	Pars_Pt.ArgVelZ = 0.0;
	Pars_Ar.Sigma = 3.40 * 1E-10;//m
	Pars_Pt.Sigma = 2.47 * 1E-10;
	Pars_Ar.Epselon = 1.654E-21;//J
	Pars_Pt.Epselon = 5.207E-20;
	Pars_Pt.Lattice = 3.93E-10;
	Pars_MD.ArPt_Sigma = 2.94 * 1E-10;
	Pars_MD.ArPt_Epselon = 1.093E-21;
	Pars_MD.CutOff = 10 * 1E-10;
	//
	Pars_Dim.Mass = Pars_Pt.Mass;
	Pars_Dim.Energy = Pars_Pt.Epselon;
	Pars_Dim.Length = Pars_Pt.Sigma;
	Pars_Dim.Velocity = sqrt(Pars_Dim.Energy / Pars_Dim.Mass);
	Pars_Dim.Time = Pars_Dim.Length / Pars_Dim.Velocity;
	Pars_Dim.Acceleration = Pars_Dim.Energy / (Pars_Dim.Mass * Pars_Dim.Length);
	//
	Pars_Ar.Mass /= Pars_Dim.Mass;
	Pars_Pt.Mass /= Pars_Dim.Mass;
	Pars_Ar.Epselon /= Pars_Dim.Energy;
	Pars_Pt.Epselon /= Pars_Dim.Energy;
	Pars_MD.ArPt_Epselon /= Pars_Dim.Energy;
	Pars_Ar.Sigma /= Pars_Dim.Length;
	Pars_Pt.Sigma /= Pars_Dim.Length;
	Pars_MD.ArPt_Sigma /= Pars_Dim.Length;
	Pars_Pt.Lattice /= Pars_Dim.Length;
	Pars_MD.CutOff /= Pars_Dim.Length;
	Pars_Ar.mpVelocity /= Pars_Dim.Velocity;
	Pars_Pt.mpVelocity /= Pars_Dim.Velocity;
	//
	Pars_MD.SprK = 5000.0;
	Pars_MD.dt = 0.001;
	Pars_MD.Rt = 100;
	Pars_MD.Tt = 3500000;
	Pars_MD.DumpStep = 100;
	Pars_MD.TimeStep = 0;
	Pars_MD.BoxXLow = 0.0;
	Pars_MD.BoxXHigh = Pt_I * Pars_Pt.Lattice;
	Pars_MD.BoxYLow = 0.0;
	Pars_MD.BoxYHigh = Pt_J * Pars_Pt.Lattice;
	Pars_MD.BoxZLow = -(Pt_K - 0.5) * Pars_Pt.Lattice;
	Pars_MD.BoxZHigh = 5.0;
	Pars_MD.State = true;
	//
	cout << "*******[Pars_Init]: Parameters Initialized!*******\n";
	cout << "Box_ZoneX: " << Pars_MD.BoxXLow << ", " << Pars_MD.BoxXHigh << "\n";
	cout << "Box_ZoneY: " << Pars_MD.BoxYLow << ", " << Pars_MD.BoxYHigh << "\n";
	cout << "Box_ZoneZ: " << Pars_MD.BoxZLow << ", " << Pars_MD.BoxZHigh << "\n";
}

/******************************************************************************/
void MD::Models_Init() {
	int count, i, j, k;
	float R1, R2, Rx, Ry;
	//
	count = 0;
	srand((unsigned)time(NULL));
	for (i = 0; i < 2 * Pt_I; i++) {
		for (j = 0; j < 2 * Pt_J; j++) {
			for (k = 0; k < 2 * Pt_K; k++) {
				if (i / 2. + j / 2. + k / 2. == int(i / 2. + j / 2. + k / 2.)) {
					Pt[count].Sid = count + 1;
					Pt[count].Lid = 1;
					Pt[count].x = i / 2.*Pars_Pt.Lattice;
					Pt[count].Bx = Pt[count].x;
					Pt[count].y = j / 2.*Pars_Pt.Lattice;
					Pt[count].By = Pt[count].y;
					Pt[count].z = (k / 2. - 2.5)*Pars_Pt.Lattice;
					Pt[count].Bz = Pt[count].z;
					R1 = random();
					R2 = random();
					Pt[count].vx = Pars_Pt.mpVelocity / sqrt(3) * sqrt(-2 * log(R1)) * cos(2 * PI * R2);
					R1 = random();
					R2 = random();
					Pt[count].vy = Pars_Pt.mpVelocity / sqrt(3) * sqrt(-2 * log(R1)) * cos(2 * PI * R2);
					R1 = random();
					R2 = random();
					Pt[count].vz = Pars_Pt.mpVelocity / sqrt(3) * sqrt(-2 * log(R1)) * cos(2 * PI * R2);
					Pt[count].ax = 0.0;
					Pt[count].ay = 0.0;
					Pt[count].az = 0.0;
					Pt[count].atx = 0.0;
					Pt[count].aty = 0.0;
					Pt[count].atz = 0.0;
					count += 1;
				}
			}
		}
	}
	//
	Ar[0].Sid = 1;
	Ar[0].Lid = 1;
	Rx = random();
	Ry = random();
	Ar[0].x = Pars_MD.BoxXLow + (Pars_MD.BoxXHigh - Pars_MD.BoxXLow) * Rx;
	Ar[0].y = Pars_MD.BoxYLow + (Pars_MD.BoxYHigh - Pars_MD.BoxYLow) * Ry;
	Ar[0].z = Pars_MD.BoxZHigh;
	R1 = random();
	R2 = random();
	Ar[0].vx = Pars_Ar.mpVelocity * sqrt(-log(R1)) * cos(2 * PI * R2);//Maxwell Distribution
	R1 = random();
	R2 = random();
	Ar[0].vy = Pars_Ar.mpVelocity * sqrt(-log(R1)) * sin(2 * PI * R2);
	R1 = random();
	Ar[0].vz = -Pars_Ar.mpVelocity * sqrt(-log(R1));
	Ar[0].ax = 0.0;
	Ar[0].ay = 0.0;
	Ar[0].az = 0.0;
	Ar[0].atx = 0.0;
	Ar[0].aty = 0.0;
	Ar[0].atz = 0.0;
	//
	cout << "*******[Models_Init]: Models Initialized!*******\n";
	cout << "Created " << Pt_N << " Pt\n";
	cout << "Created " << Ar_N << " Ar\n";
	cout << "Ar Incidence Speed: " << Ar[0].vx << "," << Ar[0].vy << "," << Ar[0].vz << "\n";
}

/******************************************************************************/
void MD::Init_Kernels() {
	//
	int i;
	Gas_Molecule *d_Ar;
	Wall_Molecule *d_Pt;
	Parameters_Gas *d_Pars_Ar;
	Parameters_Wall *d_Pars_Pt;
	Parameters_MD *d_Pars_MD;
	cudaError_t cudaStatus;
	//
	__global__ void Boundary_XY(Gas_Molecule *Ar, Wall_Molecule *Pt, Parameters_MD *Pars_MD);
	__global__ void RescaleT2(Wall_Molecule *Pt, Parameters_Wall *Pars_Pt);
	__global__ void RescaleT4(Wall_Molecule *Pt, Parameters_Wall *Pars_Pt);
	__global__ void AccelerationCal(Gas_Molecule *Ar, Wall_Molecule *Pt, Parameters_Gas *Pars_Ar, Parameters_Wall *Pars_Pt, Parameters_MD *Pars_MD);
	//
	cudaMalloc((void**)&d_Ar, sizeof(Ar));
	cudaMalloc((void**)&d_Pt, sizeof(Pt));
	cudaMalloc((void**)&d_Pars_Ar, sizeof(Parameters_Gas));
	cudaMalloc((void**)&d_Pars_Pt, sizeof(Parameters_Wall));
	cudaMalloc((void**)&d_Pars_MD, sizeof(Parameters_MD));
	//
	cudaMemcpy(d_Ar, Ar, sizeof(Ar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt, Pt, sizeof(Pt), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pars_Ar, &Pars_Ar, sizeof(Parameters_Gas), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pars_Pt, &Pars_Pt, sizeof(Parameters_Wall), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pars_MD, &Pars_MD, sizeof(Parameters_MD), cudaMemcpyHostToDevice);
	//
	Boundary_XY << <1, Pt_N + Ar_N >> >(d_Ar, d_Pt, d_Pars_MD);
	cudaDeviceSynchronize();
	//
	Pars_MD.BoxZLow = Pt[0].z;
	for (i = 0; i<Pt_N; i++) {
		if (Pt[i].z<Pars_MD.BoxZLow) {
			Pars_MD.BoxZLow = Pt[i].z;
		}
	}
	cout << "Box Z Low: " << Pars_MD.BoxZLow << "\n";
	cudaMemcpy(d_Pars_MD, &Pars_MD, sizeof(Parameters_MD), cudaMemcpyHostToDevice);
	//
	RescaleT1();
	cudaMemcpy(d_Pars_Pt, &Pars_Pt, sizeof(Parameters_Wall), cudaMemcpyHostToDevice);
	RescaleT2 << <1, Pt_N >> > (d_Pt, d_Pars_Pt);
	cudaDeviceSynchronize();
	cudaMemcpy(Pt, d_Pt, sizeof(Pt), cudaMemcpyDeviceToHost);
	RescaleT3();
	cudaMemcpy(d_Pars_Pt, &Pars_Pt, sizeof(Parameters_Wall), cudaMemcpyHostToDevice);
	RescaleT4 << <1, Pt_N >> > (d_Pt, d_Pars_Pt);
	cudaDeviceSynchronize();
	//
	AccelerationCal << <1, Pt_N + Ar_N >> >(d_Ar, d_Pt, d_Pars_Ar, d_Pars_Pt, d_Pars_MD);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "AccelerationCal launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();
	//
	cudaMemcpy(Ar, d_Ar, sizeof(Ar), cudaMemcpyDeviceToHost);
	cudaMemcpy(Pt, d_Pt, sizeof(Pt), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Pars_Ar, d_Pars_Ar, sizeof(Parameters_Gas), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Pars_Pt, d_Pars_Pt, sizeof(Parameters_Wall), cudaMemcpyDeviceToHost);
	cudaMemcpy(&Pars_MD, d_Pars_MD, sizeof(Parameters_MD), cudaMemcpyDeviceToHost);
	//
	cudaFree(d_Ar);
	cudaFree(d_Pt);
	cudaFree(d_Pars_Ar);
	cudaFree(d_Pars_Pt);
	cudaFree(d_Pars_MD);
	//
	Exit();
	//
	cout << "*******[Init_Kernels]: Initialization Done!*******\n";
	cout << "Pt Average Speed in X: " << Pars_Pt.ArgVelX << "\n";
	cout << "Pt Average Speed in Y: " << Pars_Pt.ArgVelY << "\n";
	cout << "Pt Average Speed in Z: " << Pars_Pt.ArgVelZ << "\n";
	cout << "Pt Temperature: " << Pars_Pt.CurrentT << "\n";
}

/******************************************************************************/
__global__ void Boundary_XY(Gas_Molecule *Ar, Wall_Molecule *Pt, Parameters_MD *Pars_MD) {
	//
	__shared__ float All_Pos[Pt_N + Ar_N][2];
	__shared__ float All_BPos[Pt_N][2];
	//
	int i;
	int tid = threadIdx.x;
	//
	if (tid<Pt_N) {
		All_Pos[tid][0] = Pt[tid].x;
		All_Pos[tid][1] = Pt[tid].y;
		All_BPos[tid][0] = Pt[tid].Bx;
		All_BPos[tid][1] = Pt[tid].By;
	}
	if (tid == Pt_N) {
		All_Pos[tid][0] = Ar[0].x;
		All_Pos[tid][1] = Ar[0].y;
	}
	//
	if (tid < Pt_N + Ar_N) {
		//X
		if (All_Pos[tid][0] < (*Pars_MD).BoxXLow) {
			All_Pos[tid][0] += (*Pars_MD).BoxXHigh - (*Pars_MD).BoxXLow;
			if (tid<Pt_N) {
				All_BPos[tid][0] += (*Pars_MD).BoxXHigh - (*Pars_MD).BoxXLow;
			}
		}
		else if (All_Pos[tid][0] >= (*Pars_MD).BoxXHigh) {
			All_Pos[tid][0] -= (*Pars_MD).BoxXHigh - (*Pars_MD).BoxXLow;
			if (tid<Pt_N) {
				All_BPos[tid][0] -= (*Pars_MD).BoxXHigh - (*Pars_MD).BoxXLow;
			}
		}
		//Y
		if (All_Pos[tid][1] < (*Pars_MD).BoxYLow) {
			All_Pos[tid][1] += (*Pars_MD).BoxYHigh - (*Pars_MD).BoxYLow;
			if (tid<Pt_N) {
				All_BPos[tid][1] += (*Pars_MD).BoxYHigh - (*Pars_MD).BoxYLow;
			}
		}
		else if (All_Pos[tid][1] >= (*Pars_MD).BoxYHigh) {
			All_Pos[tid][1] -= (*Pars_MD).BoxYHigh - (*Pars_MD).BoxYLow;
			if (tid<Pt_N) {
				All_BPos[tid][1] -= (*Pars_MD).BoxYHigh - (*Pars_MD).BoxYLow;
			}
		}
	}
	//
	if (tid<Pt_N) {
		Pt[tid].x = All_Pos[tid][0];
		Pt[tid].y = All_Pos[tid][1];
		Pt[tid].Bx = All_BPos[tid][0];
		Pt[tid].By = All_BPos[tid][1];
	}
	if (tid == Pt_N) {
		Ar[0].x = All_Pos[tid][0];
		Ar[0].y = All_Pos[tid][1];
	}
}

/******************************************************************************/
void MD::RescaleT1() {
	//
	int i;
	//
	Pars_Pt.ArgVelX = 0.0;
	Pars_Pt.ArgVelY = 0.0;
	Pars_Pt.ArgVelZ = 0.0;
	for (i = 0; i < Pt_N; i++) {
		Pars_Pt.ArgVelX += Pt[i].vx / Pt_N;
		Pars_Pt.ArgVelY += Pt[i].vy / Pt_N;
		Pars_Pt.ArgVelZ += Pt[i].vz / Pt_N;
	}
}

/******************************************************************************/
__global__ void RescaleT2(Wall_Molecule *Pt, Parameters_Wall *Pars_Pt) {
	//
	__shared__ float Pt_Vel[Pt_N][3];
	//
	int tid = threadIdx.x;
	//
	if (tid<Pt_N) {
		Pt_Vel[tid][0] = Pt[tid].vx;
		Pt_Vel[tid][1] = Pt[tid].vy;
		Pt_Vel[tid][2] = Pt[tid].vz;
	}
	//
	if (tid<Pt_N) {
		Pt_Vel[tid][0] -= (*Pars_Pt).ArgVelX;
		Pt_Vel[tid][1] -= (*Pars_Pt).ArgVelY;
		Pt_Vel[tid][2] -= (*Pars_Pt).ArgVelZ;
	}
	//
	if (tid<Pt_N) {
		Pt[tid].vx = Pt_Vel[tid][0];
		Pt[tid].vy = Pt_Vel[tid][1];
		Pt[tid].vz = Pt_Vel[tid][2];
	}
}

/******************************************************************************/
void MD::RescaleT3() {
	//
	int i;
	//
	Pars_Pt.CurrentT = 0.0;
	for (i = 0; i < Pt_N; i++) {
		Pars_Pt.CurrentT += Pt[i].vx * Pt[i].vx + Pt[i].vy * Pt[i].vy + Pt[i].vz * Pt[i].vz;
	}
	Pars_Pt.CurrentT *= Pars_Dim.Velocity * Pars_Dim.Velocity * Pars_Pt.Mass * Pars_Dim.Mass / (3 * Pt_N * kB);
}

/******************************************************************************/
__global__ void RescaleT4(Wall_Molecule *Pt, Parameters_Wall *Pars_Pt) {
	//
	__shared__ float Pt_Vel[Pt_N][3];
	//
	int tid = threadIdx.x;
	//
	if (tid<Pt_N) {
		Pt_Vel[tid][0] = Pt[tid].vx;
		Pt_Vel[tid][1] = Pt[tid].vy;
		Pt_Vel[tid][2] = Pt[tid].vz;
	}
	//
	if (tid<Pt_N) {
		Pt_Vel[tid][0] *= sqrt((*Pars_Pt).T / (*Pars_Pt).CurrentT);
		Pt_Vel[tid][1] *= sqrt((*Pars_Pt).T / (*Pars_Pt).CurrentT);
		Pt_Vel[tid][2] *= sqrt((*Pars_Pt).T / (*Pars_Pt).CurrentT);
	}
	//
	if (tid<Pt_N) {
		Pt[tid].vx = Pt_Vel[tid][0];
		Pt[tid].vy = Pt_Vel[tid][1];
		Pt[tid].vz = Pt_Vel[tid][2];
	}
}

/******************************************************************************/
__global__ void AccelerationCal(Gas_Molecule *Ar, Wall_Molecule *Pt, Parameters_Gas *Pars_Ar, Parameters_Wall *Pars_Pt, Parameters_MD *Pars_MD) {
	//
	__shared__ float All_Pos[Pt_N + Ar_N][3];
	__shared__ float All_BPos[Pt_N][3];
	__shared__ float All_Acc[Pt_N + Ar_N][3];
	//
	int tid = threadIdx.x;
	int i;
	float Epair, Spair, Pairx, Pairy, Pairz, Dispair, Fpair, Atom_Fx, Atom_Fy, Atom_Fz;
	float Spring_Disx, Spring_Fx, Pt_Fx, Spring_Disy, Spring_Fy, Pt_Fy, Spring_Disz, Spring_Fz, Pt_Fz, Ar_Fx, Ar_Fy, Ar_Fz;
	//
	if (tid<Pt_N) {
		//
		All_Pos[tid][0] = Pt[tid].x;
		All_Pos[tid][1] = Pt[tid].y;
		All_Pos[tid][2] = Pt[tid].z;
		All_BPos[tid][0] = Pt[tid].Bx;
		All_BPos[tid][1] = Pt[tid].By;
		All_BPos[tid][2] = Pt[tid].Bz;
		All_Acc[tid][0] = 0.0;
		All_Acc[tid][1] = 0.0;
		All_Acc[tid][2] = 0.0;
	}
	if (tid == Pt_N) {
		//
		All_Pos[tid][0] = Ar[0].x;
		All_Pos[tid][1] = Ar[0].y;
		All_Pos[tid][2] = Ar[0].z;
		All_Acc[tid][0] = 0.0;
		All_Acc[tid][1] = 0.0;
		All_Acc[tid][2] = 0.0;
	}
	__syncthreads();
	//
	if (tid<Pt_N + Ar_N) {
		Atom_Fx = 0.0;
		Atom_Fy = 0.0;
		Atom_Fz = 0.0;
		for (i = 0; i < Pt_N + Ar_N; i++) {
			if (tid<Pt_N&&i<Pt_N) {
				Epair = (*Pars_Pt).Epselon;
				Spair = (*Pars_Pt).Sigma;
			}
			else if (tid == Pt_N && i == Pt_N) {
				Epair = (*Pars_Ar).Epselon;
				Spair = (*Pars_Ar).Sigma;
			}
			else {
				Epair = (*Pars_MD).ArPt_Epselon;
				Spair = (*Pars_MD).ArPt_Sigma;
			}
			Pairx = All_Pos[tid][0] - All_Pos[i][0];
			Pairy = All_Pos[tid][1] - All_Pos[i][1];
			Pairz = All_Pos[tid][2] - All_Pos[i][2];
			//
			if (abs(Pairx) >= (*Pars_MD).BoxXHigh - (*Pars_MD).BoxXLow - (*Pars_MD).CutOff) {
				Pairx -= ((*Pars_MD).BoxXHigh - (*Pars_MD).BoxXLow)*Pairx / abs(Pairx);
			}
			if (abs(Pairy) >= (*Pars_MD).BoxYHigh - (*Pars_MD).BoxYLow - (*Pars_MD).CutOff) {
				Pairy -= ((*Pars_MD).BoxYHigh - (*Pars_MD).BoxYLow)*Pairy / abs(Pairy);
			}
			//
			Dispair = sqrt(Pairx * Pairx + Pairy * Pairy + Pairz * Pairz);
			if (Dispair > 0 && Dispair <= (*Pars_MD).CutOff) {
				Fpair = 48 * Epair*(pow(Spair, 12) / pow(Dispair, 13) - 0.5*pow(Spair, 6) / pow(Dispair, 7));
				Atom_Fx += Pairx * Fpair / Dispair;
				Atom_Fy += Pairy * Fpair / Dispair;
				Atom_Fz += Pairz * Fpair / Dispair;
			}
		}
		if (tid<Pt_N) {
			//
			Spring_Disx = All_Pos[tid][0] - All_BPos[tid][0];
			Spring_Fx = -(*Pars_MD).SprK * Spring_Disx;
			Pt_Fx = Atom_Fx + Spring_Fx;
			All_Acc[tid][0] = Pt_Fx / (*Pars_Pt).Mass;
			Spring_Disy = All_Pos[tid][1] - All_BPos[tid][1];
			Spring_Fy = -(*Pars_MD).SprK * Spring_Disy;
			Pt_Fy = Atom_Fy + Spring_Fy;
			All_Acc[tid][1] = Pt_Fy / (*Pars_Pt).Mass;
			Spring_Disz = All_Pos[tid][2] - All_BPos[tid][2];
			Spring_Fz = -(*Pars_MD).SprK * Spring_Disz;
			Pt_Fz = Atom_Fz + Spring_Fz;
			All_Acc[tid][2] = Pt_Fz / (*Pars_Pt).Mass;
		}
		else {
			//
			Ar_Fx = Atom_Fx;
			All_Acc[tid][0] = Ar_Fx / (*Pars_Ar).Mass;
			Ar_Fy = Atom_Fy;
			All_Acc[tid][1] = Ar_Fy / (*Pars_Ar).Mass;
			Ar_Fz = Atom_Fz;
			All_Acc[tid][2] = Ar_Fz / (*Pars_Ar).Mass;
		}
	}
	//
	if (tid<Pt_N) {
		//
		Pt[tid].ax = All_Acc[tid][0];
		Pt[tid].ay = All_Acc[tid][1];
		Pt[tid].az = All_Acc[tid][2];
	}
	if (tid == Pt_N) {
		//
		Ar[0].ax = All_Acc[tid][0];
		Ar[0].ay = All_Acc[tid][1];
		Ar[0].az = All_Acc[tid][2];
	}
}

/******************************************************************************/
__global__ void Verlet_Pos(Gas_Molecule *Ar, Wall_Molecule *Pt, Parameters_MD *Pars_MD) {
	//
	__shared__ float All_Pos[Pt_N + Ar_N][3];
	__shared__ float All_Vel[Pt_N + Ar_N][3];
	__shared__ float All_Acc[Pt_N + Ar_N][3];
	//
	int tid = threadIdx.x;
	//
	if (tid<Pt_N) {
		//
		All_Pos[tid][0] = Pt[tid].x;
		All_Pos[tid][1] = Pt[tid].y;
		All_Pos[tid][2] = Pt[tid].z;
		All_Vel[tid][0] = Pt[tid].vx;
		All_Vel[tid][1] = Pt[tid].vy;
		All_Vel[tid][2] = Pt[tid].vz;
		All_Acc[tid][0] = Pt[tid].ax;
		All_Acc[tid][1] = Pt[tid].ay;
		All_Acc[tid][2] = Pt[tid].az;
	}
	if (tid == Pt_N) {
		//
		All_Pos[tid][0] = Ar[0].x;
		All_Pos[tid][1] = Ar[0].y;
		All_Pos[tid][2] = Ar[0].z;
		All_Vel[tid][0] = Ar[0].vx;
		All_Vel[tid][1] = Ar[0].vy;
		All_Vel[tid][2] = Ar[0].vz;
		All_Acc[tid][0] = Ar[0].ax;
		All_Acc[tid][1] = Ar[0].ay;
		All_Acc[tid][2] = Ar[0].az;
	}
	//
	if (tid<Pt_N + Ar_N) {
		All_Pos[tid][0] += All_Vel[tid][0] * (*Pars_MD).dt + 0.5 * All_Acc[tid][0] * (*Pars_MD).dt * (*Pars_MD).dt;
		All_Pos[tid][1] += All_Vel[tid][1] * (*Pars_MD).dt + 0.5 * All_Acc[tid][1] * (*Pars_MD).dt * (*Pars_MD).dt;
		All_Pos[tid][2] += All_Vel[tid][2] * (*Pars_MD).dt + 0.5 * All_Acc[tid][2] * (*Pars_MD).dt * (*Pars_MD).dt;
	}
	//
	if (tid<Pt_N) {
		//
		Pt[tid].x = All_Pos[tid][0];
		Pt[tid].y = All_Pos[tid][1];
		Pt[tid].z = All_Pos[tid][2];
	}
	if (tid == Pt_N) {
		//
		Ar[0].x = All_Pos[tid][0];
		Ar[0].y = All_Pos[tid][1];
		Ar[0].z = All_Pos[tid][2];
	}
}

/******************************************************************************/
__global__ void Verlet_Vel(Gas_Molecule *Ar, Wall_Molecule *Pt, Parameters_MD *Pars_MD) {
	//
	__shared__ float All_Vel[Pt_N + Ar_N][3];
	__shared__ float All_Acc[Pt_N + Ar_N][3];
	__shared__ float All_Acc_Temp[Pt_N + Ar_N][3];
	//
	int tid = threadIdx.x;
	//
	if (tid<Pt_N) {
		//
		All_Vel[tid][0] = Pt[tid].vx;
		All_Vel[tid][1] = Pt[tid].vy;
		All_Vel[tid][2] = Pt[tid].vz;
		All_Acc[tid][0] = Pt[tid].ax;
		All_Acc[tid][1] = Pt[tid].ay;
		All_Acc[tid][2] = Pt[tid].az;
		All_Acc_Temp[tid][0] = Pt[tid].atx;
		All_Acc_Temp[tid][1] = Pt[tid].aty;
		All_Acc_Temp[tid][2] = Pt[tid].atz;
	}
	if (tid == Pt_N) {
		//
		All_Vel[tid][0] = Ar[0].vx;
		All_Vel[tid][1] = Ar[0].vy;
		All_Vel[tid][2] = Ar[0].vz;
		All_Acc[tid][0] = Ar[0].ax;
		All_Acc[tid][1] = Ar[0].ay;
		All_Acc[tid][2] = Ar[0].az;
		All_Acc_Temp[tid][0] = Ar[0].atx;
		All_Acc_Temp[tid][1] = Ar[0].aty;
		All_Acc_Temp[tid][2] = Ar[0].atz;
	}
	//
	if (tid<Pt_N + Ar_N) {
		All_Vel[tid][0] += 0.5 * (All_Acc_Temp[tid][0] + All_Acc[tid][0]) * (*Pars_MD).dt;
		All_Vel[tid][1] += 0.5 * (All_Acc_Temp[tid][1] + All_Acc[tid][1]) * (*Pars_MD).dt;
		All_Vel[tid][2] += 0.5 * (All_Acc_Temp[tid][2] + All_Acc[tid][2]) * (*Pars_MD).dt;
	}
	//
	if (tid < Pt_N) {
		Pt[tid].vx = All_Vel[tid][0];
		Pt[tid].vy = All_Vel[tid][1];
		Pt[tid].vz = All_Vel[tid][2];
	}
	if (tid == Pt_N) {
		Ar[0].vx = All_Vel[tid][0];
		Ar[0].vy = All_Vel[tid][1];
		Ar[0].vz = All_Vel[tid][2];
	}
}

/******************************************************************************/
__global__ void Last_Acceleration(Gas_Molecule *Ar, Wall_Molecule *Pt) {
	//
	__shared__ float All_Acc[Pt_N + Ar_N][3];
	__shared__ float All_Acc_Temp[Pt_N + Ar_N][3];
	//
	int tid = threadIdx.x;
	//
	if (tid<Pt_N) {
		//
		All_Acc[tid][0] = Pt[tid].ax;
		All_Acc[tid][1] = Pt[tid].ay;
		All_Acc[tid][2] = Pt[tid].az;
		All_Acc_Temp[tid][0] = Pt[tid].atx;
		All_Acc_Temp[tid][1] = Pt[tid].aty;
		All_Acc_Temp[tid][2] = Pt[tid].atz;
	}
	if (tid == Pt_N) {
		//
		All_Acc[tid][0] = Ar[0].ax;
		All_Acc[tid][1] = Ar[0].ay;
		All_Acc[tid][2] = Ar[0].az;
		All_Acc_Temp[tid][0] = Ar[0].atx;
		All_Acc_Temp[tid][1] = Ar[0].aty;
		All_Acc_Temp[tid][2] = Ar[0].atz;
	}
	//
	if (tid<Pt_N + Ar_N) {
		All_Acc_Temp[tid][0] = All_Acc[tid][0];
		All_Acc_Temp[tid][1] = All_Acc[tid][1];
		All_Acc_Temp[tid][2] = All_Acc[tid][2];
	}
	//
	if (tid < Pt_N) {
		Pt[tid].atx = All_Acc_Temp[tid][0];
		Pt[tid].aty = All_Acc_Temp[tid][1];
		Pt[tid].atz = All_Acc_Temp[tid][2];
	}
	if (tid == Pt_N) {
		Ar[0].atx = All_Acc_Temp[tid][0];
		Ar[0].aty = All_Acc_Temp[tid][1];
		Ar[0].atz = All_Acc_Temp[tid][2];
	}
}

/******************************************************************************/
void MD::Dump() {
	//
	int i;
	//
	if (Pars_MD.TimeStep % Pars_MD.DumpStep == 0) {
		//
		ofstream MD;
		MD.open("Kernel_MD_CUDA_C.dump", ios::app);
		MD << "ITEM: TIMESTEP\n";
		MD << Pars_MD.TimeStep << "\n";
		MD << "ITEM: NUMBER OF ATOMS\n";
		MD << Pt_N + Ar_N << "\n";
		MD << "ITEM: BOX BOUNDS pp pp ff\n";
		MD << Pars_MD.BoxXLow << " " << Pars_MD.BoxXHigh << "\n";
		MD << Pars_MD.BoxYLow << " " << Pars_MD.BoxYHigh << "\n";
		MD << Pars_MD.BoxZLow << " " << Pars_MD.BoxZHigh << "\n";
		MD << "ITEM: ATOMS id type x y z\n";
		for (i = 0; i < Pt_N; i++) {
			MD << i + 1 << " " << Pars_Pt.Type << " " << Pt[i].x << " " << Pt[i].y << " " << Pt[i].z << "\n";
		}
		MD << Pt_N + Ar_N << " " << Pars_Ar.Type << " " << Ar[0].x << " " << Ar[0].y << " " << Ar[0].z << "\n";
		MD.close();
		//
		ofstream Zt;
		Zt.open("Kernel_MD_CUDA_C_Zt.dat", ios::app);
		Zt << Pars_MD.TimeStep * Pars_MD.dt << " " << Ar[0].z << "\n";
		Zt.close();
	}
}

/******************************************************************************/
void MD::Exit() {
	//
	if (Ar[0].z > Pars_MD.BoxZHigh || Pars_MD.TimeStep >= Pars_MD.Tt) {
		Pars_MD.State = false;
		Pars_MD.DumpStep = 1;
		Dump();
	}
	else {
		Dump();
	}
}

/******************************************************************************/
float MD::random() {
	//
	float R;
	//
	R = 0.0;
	while (R == 0.0) {
		R = rand() / float(RAND_MAX);
	}
	return R;
}

/******************************************************************************/
void MD::MainMD() {
	//
	int i;
	clock_t start, finish;
	float tperl;
	Gas_Molecule *d_Ar;
	Wall_Molecule *d_Pt;
	Parameters_Gas *d_Pars_Ar;
	Parameters_Wall *d_Pars_Pt;
	Parameters_MD *d_Pars_MD;
	//
	cudaMalloc((void**)&d_Ar, sizeof(Ar));
	cudaMalloc((void**)&d_Pt, sizeof(Pt));
	cudaMalloc((void**)&d_Pars_Ar, sizeof(Parameters_Gas));
	cudaMalloc((void**)&d_Pars_Pt, sizeof(Parameters_Wall));
	cudaMalloc((void**)&d_Pars_MD, sizeof(Parameters_MD));
	//
	cudaMemcpy(d_Ar, Ar, sizeof(Ar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt, Pt, sizeof(Pt), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pars_Ar, &Pars_Ar, sizeof(Parameters_Gas), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pars_Pt, &Pars_Pt, sizeof(Parameters_Wall), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pars_MD, &Pars_MD, sizeof(Parameters_MD), cudaMemcpyHostToDevice);
	//
	start = clock();
	while (Pars_MD.State) {
		//
		Verlet_Pos << <1, Pt_N + Ar_N >> >(d_Ar, d_Pt, d_Pars_MD);
		cudaDeviceSynchronize();
		//
		Boundary_XY << <1, Pt_N + Ar_N >> >(d_Ar, d_Pt, d_Pars_MD);
		cudaDeviceSynchronize();
		//
		Last_Acceleration << <1, Pt_N + Ar_N >> >(d_Ar, d_Pt);
		cudaDeviceSynchronize();
		//
		AccelerationCal << <1, Pt_N + Ar_N >> >(d_Ar, d_Pt, d_Pars_Ar, d_Pars_Pt, d_Pars_MD);
		cudaDeviceSynchronize();
		//
		Verlet_Vel << <1, Pt_N + Ar_N >> >(d_Ar, d_Pt, d_Pars_MD);
		cudaDeviceSynchronize();
		//
		cudaMemcpy(Pt, d_Pt, sizeof(Pt), cudaMemcpyDeviceToHost);
		RescaleT1();
		cudaMemcpy(d_Pars_Pt, &Pars_Pt, sizeof(Parameters_Wall), cudaMemcpyHostToDevice);
		RescaleT2 << <1, Pt_N >> > (d_Pt, d_Pars_Pt);
		cudaDeviceSynchronize();
		cudaMemcpy(Pt, d_Pt, sizeof(Pt), cudaMemcpyDeviceToHost);
		RescaleT3();
		cudaMemcpy(d_Pars_Pt, &Pars_Pt, sizeof(Parameters_Wall), cudaMemcpyHostToDevice);
		RescaleT4 << <1, Pt_N >> > (d_Pt, d_Pars_Pt);
		cudaDeviceSynchronize();
		//
		cudaMemcpy(Ar, d_Ar, sizeof(Ar), cudaMemcpyDeviceToHost);
		cudaMemcpy(Pt, d_Pt, sizeof(Pt), cudaMemcpyDeviceToHost);
		//
		Pars_MD.BoxZLow = Pt[0].z;
		for (i = 0; i<Pt_N; i++) {
			if (Pt[i].z<Pars_MD.BoxZLow) {
				Pars_MD.BoxZLow = Pt[i].z;
			}
		}
		//
		Pars_MD.TimeStep += 1;
		Exit();
		finish = clock();
		tperl = float(finish - start) / CLOCKS_PER_SEC / Pars_MD.TimeStep;
		cout << Pars_MD.TimeStep << " TimeSteps; ArgTime: " << tperl << " Seconds!\r";
	}
	//
	cudaFree(d_Ar);
	cudaFree(d_Pt);
	cudaFree(d_Pars_Ar);
	cudaFree(d_Pars_Pt);
	cudaFree(d_Pars_MD);
	//
	cout << "\n*******[MainMD]: MD Done!*******\n";
}

////////////////////////////////////////////////////////////////////////////////
/*************************************main*************************************/
////////////////////////////////////////////////////////////////////////////////
int main() {
	//
	class MD GasWall;
	GasWall.Pars_Init();
	GasWall.Models_Init();
	GasWall.Init_Kernels();
	GasWall.MainMD();
	system("pause");
	return 0;
}
