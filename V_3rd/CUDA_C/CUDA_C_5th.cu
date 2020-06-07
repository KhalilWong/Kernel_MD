#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <algorithm>
using namespace std;
const int Lib_N = 100000;
const int SampleParallel_N = 1000;
const int Atom_type_N = 2;
const int Pair_type_N = 3;
const int Pt_I = 6, Pt_J = 6, Pt_K = 3;
const int Pt_N = 4 * Pt_I*Pt_J*Pt_K;
const int Ar_N = 1;

class Parameters {
public:
	int Rt, Tt, dumpstep;
	double Mass[Atom_type_N], T[Atom_type_N], mp_V[Atom_type_N], LJ_E[Pair_type_N], LJ_S[Pair_type_N], Box[3][2], Pt_ePos[Pt_N][3];
	double PI, kB, fcc_lattice, nd_Mass, nd_Energy, nd_Length, nd_Velocity, nd_Time, nd_Acceleration, cutoff, d, spr_k, dt, Pt_argVelx, Pt_argVely, Pt_argVelz, Pt_T;
	bool state;
	void Init();
	void Initialization(int All_type[], double All_Pos[][3], double All_Vel[][3], double All_Acc[][3]);
	void Initialization_Kernel(int All_type[], double All_Pos[][3], double All_Vel[][3], double All_Acc[][3]);
	void rescale_T1(double All_Vel[][3]);
	void rescale_T3(double All_Vel[][3]);
	void Dump(int All_type[], double All_Pos[][3], int timestep, int ds = 1);
	void Exit(int All_type[], double All_Pos[][3], int timestep);
	double random();
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
	//盒子状态
	Box[0][0] = 0;
	Box[0][1] = Pt_I * fcc_lattice;
	Box[1][0] = 0;
	Box[1][1] = Pt_J * fcc_lattice;
	Box[2][0] = -(Pt_K - 0.5)*fcc_lattice;
	Box[2][1] = d;
	//状态参数
	state = true;

	cout << "*******Parameters Initialized!*******\n";
}

/******************************************************************************/
void Parameters::Initialization(int All_type[], double All_Pos[][3], double All_Vel[][3], double All_Acc[][3]) {
	double *d_Pt_argVelx, *d_Pt_argVely, *d_Pt_argVelz, *d_Pt_T, *d_Mass, *d_T, *d_LJ_E, *d_LJ_S, *d_cutoff, *d_spr_k;
	int i;
	double(*d_All_Pos)[3], (*d_All_Vel)[3], (*d_All_Acc)[3], (*d_Box)[2], (*d_Pt_ePos)[3];
	__global__ void Pos_period(double(*All_Pos)[3], double(*Box)[2], double(*Pt_ePos)[3]);
	__global__ void rescale_T2(double(*All_Vel)[3], double *Pt_argVelx, double *Pt_argVely, double *Pt_argVelz);
	__global__ void rescale_T4(double(*All_Vel)[3], double *Pt_T, double *T);
	//__global__ void rescale_T(double All_Vel[][3], double *Pt_argVelx, double *Pt_argVely, double *Pt_argVelz, double *Pt_V2, double *nd_Velocity, double Mass[2], double *nd_Mass, double *kB, double T[2]);
	__global__ void Acceleration_period(double(*All_Pos)[3], double(*All_Acc)[3], double *LJ_E, double *LJ_S, double(*Box)[2], double *cutoff, double(*Pt_ePos)[3], double *spr_k, double *Mass);

	cout << "计算区域X: " << Box[0][0] << ", " << Box[0][1] << "\n";
	cout << "计算区域Y: " << Box[1][0] << ", " << Box[1][1] << "\n";
	cout << "计算区域Z: " << Box[2][0] << ", " << Box[2][1] << "\n";
	//位置，速度初始化
	Initialization_Kernel(All_type, All_Pos, All_Vel, All_Acc);
	//分配内存，初始化
	cudaMalloc((void**)&d_All_Pos, sizeof(All_Pos));
	cudaMalloc((void**)&d_All_Vel, sizeof(All_Vel));
	cudaMalloc((void**)&d_All_Acc, sizeof(All_Acc));
	cudaMalloc((void**)&d_Box, sizeof(Box));
	cudaMalloc((void**)&d_Pt_ePos, sizeof(Pt_ePos));
	cudaMalloc((void**)&d_Mass, sizeof(Mass));
	cudaMalloc((void**)&d_T, sizeof(T));
	cudaMalloc((void**)&d_LJ_E, sizeof(LJ_E));
	cudaMalloc((void**)&d_LJ_S, sizeof(LJ_S));
	cudaMalloc((void**)&d_cutoff, sizeof(double));
	cudaMalloc((void**)&d_spr_k, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVelx, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVely, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVelz, sizeof(double));
	cudaMalloc((void**)&d_Pt_T, sizeof(double));
	cudaMemcpy(d_All_Pos, All_Pos, sizeof(All_Pos), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Vel, All_Vel, sizeof(All_Vel), cudaMemcpyHostToDevice);
	cudaMemcpy(d_All_Acc, All_Acc, sizeof(All_Acc), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box, Box, sizeof(Box), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_ePos, Pt_ePos, sizeof(Pt_ePos), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mass, Mass, sizeof(Mass), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, T, sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_E, LJ_E, sizeof(LJ_E), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_S, LJ_S, sizeof(LJ_S), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cutoff, &cutoff, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_spr_k, &spr_k, sizeof(double), cudaMemcpyHostToDevice);
	//首次位置周期
	Pos_period << <1, Pt_N + Ar_N >> >(d_All_Pos, d_Box, d_Pt_ePos);
	cudaDeviceSynchronize();
	cudaMemcpy(All_Pos, d_All_Pos, sizeof(All_Pos), cudaMemcpyDeviceToHost);
	Box[2][0] = All_Pos[0][2];
	for (i = 0; i < Pt_N; i++) {
		if (All_Pos[i][2] < Box[2][0]) {
			Box[2][0] = All_Pos[i][2];
		}
		//Pt_ePos[i][0] = All_Pos[i][0];
		//Pt_ePos[i][1] = All_Pos[i][1];
		//Pt_ePos[i][2] = All_Pos[i][2];
	}
	//cudaMemcpy(d_Pt_ePos, Pt_ePos, sizeof(Pt_ePos), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box, Box, sizeof(Box), cudaMemcpyHostToDevice);
	//首次控温
	rescale_T1(All_Vel);
	cudaMemcpy(d_Pt_argVelx, &Pt_argVelx, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_argVely, &Pt_argVely, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt_argVelz, &Pt_argVelz, sizeof(double), cudaMemcpyHostToDevice);
	rescale_T2 << <1, Pt_N >> > (d_All_Vel, d_Pt_argVelx, d_Pt_argVely, d_Pt_argVelz);
	cudaDeviceSynchronize();
	cudaMemcpy(All_Vel, d_All_Vel, sizeof(All_Vel), cudaMemcpyDeviceToHost);
	rescale_T3(All_Vel);
	cudaMemcpy(d_Pt_T, &Pt_T, sizeof(double), cudaMemcpyHostToDevice);
	rescale_T4 << <1, Pt_N >> > (d_All_Vel, d_Pt_T, d_T);
	cudaDeviceSynchronize();
	//首次加速度周期
	Acceleration_period << <1, Pt_N + Ar_N >> >(d_All_Pos, d_All_Acc, d_LJ_E, d_LJ_S, d_Box, d_cutoff, d_Pt_ePos, d_spr_k, d_Mass);
	cudaDeviceSynchronize();
	cudaMemcpy(All_Pos, d_All_Pos, sizeof(All_Pos), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Vel, d_All_Vel, sizeof(All_Vel), cudaMemcpyDeviceToHost);
	cudaMemcpy(All_Acc, d_All_Acc, sizeof(All_Acc), cudaMemcpyDeviceToHost);
	//初始信息
	cout << "Created " << Pt_N << " Pt\n";
	cout << "Created " << Ar_N << " Ar\n";
	cout << "Pt整体x方向平均速度" << Pt_argVelx << "\n";
	cout << "Pt整体y方向平均速度" << Pt_argVely << "\n";
	cout << "Pt整体z方向平均速度" << Pt_argVelz << "\n";
	cout << "Pt温度" << Pt_T << "\n";
	cout << "Ar入射速度:" << All_Vel[Pt_N][0] << "," << All_Vel[Pt_N][1] << "," << All_Vel[Pt_N][2] << "\n";
	cout << "*******Model Initialization Done!*******\n";
	cudaFree(d_All_Pos);
	cudaFree(d_All_Vel);
	cudaFree(d_All_Acc);
	cudaFree(d_Box);
	cudaFree(d_Pt_ePos);
	cudaFree(d_Mass);
	cudaFree(d_T);
	cudaFree(d_LJ_E);
	cudaFree(d_LJ_S);
	cudaFree(d_cutoff);
	cudaFree(d_spr_k);
	cudaFree(d_Pt_argVelx);
	cudaFree(d_Pt_argVely);
	cudaFree(d_Pt_argVelz);
	cudaFree(d_Pt_T);
}

/******************************************************************************/
void Parameters::Initialization_Kernel(int All_type[], double All_Pos[][3], double All_Vel[][3],double All_Acc[][3]) {
	int i, j, k, axis, count;
	double R1, R2, Rx, Ry;

	count = 0;
	srand((unsigned)time(NULL));
	for (i = 0; i < 2 * Pt_I; i++) {
		for (j = 0; j < 2 * Pt_J; j++) {
			for (k = 0; k < 2 * Pt_K; k++) {
				if (i / 2. + j / 2. + k / 2. == int(i / 2. + j / 2. + k / 2.)) {
					All_type[count] = 1;
					All_Pos[count][0] = i / 2.*fcc_lattice;
					Pt_ePos[count][0] = All_Pos[count][0];
					All_Pos[count][1] = j / 2.*fcc_lattice;
					Pt_ePos[count][1] = All_Pos[count][1];
					All_Pos[count][2] = (k / 2. - 2.5)*fcc_lattice;
					Pt_ePos[count][2] = All_Pos[count][2];
					for (axis = 0; axis < 3; axis++) {
						R1 = random();
						R2 = random();
						All_Vel[count][axis] = mp_V[1] / sqrt(3)*sqrt(-2 * log(R1))*cos(2 * PI*R2);
					}
					All_Acc[count][0] = 0.0;
					All_Acc[count][1] = 0.0;
					All_Acc[count][2] = 0.0;
					count += 1;
				}
			}
		}
	}
	Rx = random();
	Ry = random();
	All_type[count] = 0;
	All_Pos[count][0] = Box[0][0] + (Box[0][1] - Box[0][0]) * Rx;
	All_Pos[count][1] = Box[1][0] + (Box[1][1] - Box[1][0]) * Ry;
	All_Pos[count][2] = Box[2][1];
	R1 = random();
	R2 = random();
	All_Vel[count][0] = mp_V[0] * sqrt(-log(R1))*cos(2 * PI*R2);//Maxwell分布
	R1 = random();
	R2 = random();
	All_Vel[count][1] = mp_V[0] * sqrt(-log(R1))*sin(2 * PI*R2);
	R1 = random();
	All_Vel[count][2] = -mp_V[0] * sqrt(-log(R1));
	All_Acc[count][0] = 0.0;
	All_Acc[count][1] = 0.0;
	All_Acc[count][2] = 0.0;
}

/******************************************************************************/
__global__ void Pos_period(double (*All_Pos)[3], double (*Box)[2], double (*Pt_ePos)[3]) {
	int axis;
	int tid = threadIdx.x;

	if (tid<Pt_N + Ar_N) {
		//X,Y方向周期
		for (axis = 0; axis<2; axis++) {
			if (All_Pos[tid][axis]<Box[axis][0]) {
				All_Pos[tid][axis] += Box[axis][1] - Box[axis][0];
				if (tid<Pt_N) {
					Pt_ePos[tid][axis] += Box[axis][1] - Box[axis][0];
				}
			}
			else if (All_Pos[tid][axis] >= Box[axis][1]) {
				All_Pos[tid][axis] -= Box[axis][1] - Box[axis][0];
				if (tid<Pt_N) {
					Pt_ePos[tid][axis] -= Box[axis][1] - Box[axis][0];
				}
			}
		}
	}
}

/******************************************************************************/
void Parameters::rescale_T1(double All_Vel[][3]) {
	int i;

	Pt_argVelx = 0.0;
	Pt_argVely = 0.0;
	Pt_argVelz = 0.0;
	for (i = 0; i < Pt_N; i++) {
		Pt_argVelx += All_Vel[i][0] / Pt_N;
		Pt_argVely += All_Vel[i][1] / Pt_N;
		Pt_argVelz += All_Vel[i][2] / Pt_N;
	}
}

/******************************************************************************/
__global__ void rescale_T2(double(*All_Vel)[3], double *Pt_argVelx, double *Pt_argVely, double *Pt_argVelz) {
	int tid = threadIdx.x;

	if (tid<Pt_N) {
		//只需要热运动速度
		All_Vel[tid][0] -= *Pt_argVelx;
		All_Vel[tid][1] -= *Pt_argVely;
		All_Vel[tid][2] -= *Pt_argVelz;
	}
}

/******************************************************************************/
void Parameters::rescale_T3(double All_Vel[][3]) {
	int i;

	Pt_T = 0.0;
	for (i = 0; i < Pt_N; i++) {
		Pt_T += All_Vel[i][0] * All_Vel[i][0] + All_Vel[i][1] * All_Vel[i][1] + All_Vel[i][2] * All_Vel[i][2];
	}
	Pt_T *= nd_Velocity * nd_Velocity * Mass[1] * nd_Mass / (3 * Pt_N*kB);
}

/******************************************************************************/
__global__ void rescale_T4(double (*All_Vel)[3], double *Pt_T, double *T) {
	int tid = threadIdx.x;

	if (tid<Pt_N) {
		All_Vel[tid][0] *= sqrt(T[1] / (*Pt_T));
		All_Vel[tid][1] *= sqrt(T[1] / (*Pt_T));
		All_Vel[tid][2] *= sqrt(T[1] / (*Pt_T));
	}
}

/******************************************************************************/
__global__ void Acceleration_period(double (*All_Pos)[3], double (*All_Acc)[3], double *LJ_E, double *LJ_S, double(*Box)[2], double *cutoff, double (*Pt_ePos)[3], double *spr_k, double *Mass) {
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
			//周期相对位置
			Pairx = All_Pos[tid][0] - All_Pos[i][0];
			Pairy = All_Pos[tid][1] - All_Pos[i][1];
			Pairz = All_Pos[tid][2] - All_Pos[i][2];
			if (abs(Pairx) >= Box[0][1] - Box[0][0] - (*cutoff)) {
				Pairx -= (Box[0][1] - Box[0][0])*Pairx / abs(Pairx);
			}
			if (abs(Pairy) >= Box[1][1] - Box[1][0] - (*cutoff)) {
				Pairy -= (Box[1][1] - Box[1][0])*Pairy / abs(Pairy);
			}
			//周期距离
			Dispair = sqrt(Pairx * Pairx + Pairy * Pairy + Pairz * Pairz);
			if (Dispair > 0 && Dispair <= (*cutoff)) {
				Fpair = 48 * Epair*(pow(Spair, 12) / pow(Dispair, 13) - 0.5*pow(Spair, 6) / pow(Dispair, 7));
				Atom_Fx += Pairx * Fpair / Dispair;
				Atom_Fy += Pairy * Fpair / Dispair;
				Atom_Fz += Pairz * Fpair / Dispair;
			}
		}
		if (tid<Pt_N) {
			//Pt弹性恢复力
			Spring_Disx = All_Pos[tid][0] - Pt_ePos[tid][0];
			Spring_Fx = -(*spr_k) * Spring_Disx;
			Pt_Fx = Atom_Fx + Spring_Fx;
			All_Acc[tid][0] = Pt_Fx / Mass[1];
			Spring_Disy = All_Pos[tid][1] - Pt_ePos[tid][1];
			Spring_Fy = -(*spr_k) * Spring_Disy;
			Pt_Fy = Atom_Fy + Spring_Fy;
			All_Acc[tid][1] = Pt_Fy / Mass[1];
			Spring_Disz = All_Pos[tid][2] - Pt_ePos[tid][2];
			Spring_Fz = -(*spr_k) * Spring_Disz;
			Pt_Fz = Atom_Fz + Spring_Fz;
			All_Acc[tid][2] = Pt_Fz / Mass[1];
		}
		else {
			//Ar
			Ar_Fx = Atom_Fx;
			All_Acc[tid][0] = Ar_Fx / Mass[0];
			Ar_Fy = Atom_Fy;
			All_Acc[tid][1] = Ar_Fy / Mass[0];
			Ar_Fz = Atom_Fz;
			All_Acc[tid][2] = Ar_Fz / Mass[0];
		}
	}
}

/******************************************************************************/
__global__ void Verlet_Pos(double(*All_Pos)[3], double(*All_Vel)[3], double(*All_Acc)[3], double *dt) {
	int axis;
	int tid = threadIdx.x;

	if (tid<Pt_N + Ar_N) {
		for (axis = 0; axis<3; axis++) {
			All_Pos[tid][axis] += All_Vel[tid][axis] * (*dt) + 0.5*All_Acc[tid][axis] * (*dt) * (*dt);
		}
	}
}

/******************************************************************************/
__global__ void Verlet_Vel(double(*All_Vel)[3], double(*All_Acc_temp)[3], double(*All_Acc)[3], double *dt) {
	int axis;
	int tid = threadIdx.x;

	if (tid<Pt_N + Ar_N) {
		for (axis = 0; axis<3; axis++) {
			All_Vel[tid][axis] += 0.5*(All_Acc_temp[tid][axis] + All_Acc[tid][axis])*(*dt);
		}
	}
}

/******************************************************************************/
void Parameters::Dump(int All_type[], double All_Pos[][3], int timestep, int ds) {
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
		for (i = 0; i < Pt_N +Ar_N; i++) {
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
void Parameters::Exit(int All_type[], double All_Pos[][3], int timestep) {

	if (All_Pos[Pt_N][2] > d || timestep >= Tt) {
		state = false;
		Dump(All_type, All_Pos, timestep);
	}
	else {
		Dump(All_type, All_Pos, timestep, dumpstep);
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
	int All_type[Pt_N + Ar_N], i;
	double All_Pos[Pt_N + Ar_N][3], All_Vel[Pt_N + Ar_N][3], All_Acc[Pt_N + Ar_N][3], All_Acc_temp[Pt_N + Ar_N][3];
	double(*d_All_Pos)[3], (*d_All_Vel)[3], (*d_All_Acc)[3], (*d_All_Acc_temp)[3], (*d_Box)[2], (*d_Pt_ePos)[3];
	double *d_dt, *d_Mass, *d_T, *d_LJ_E, *d_LJ_S, *d_cutoff, *d_spr_k, *d_Pt_argVelx, *d_Pt_argVely, *d_Pt_argVelz, *d_Pt_T;
	int timestep;
	__global__ void Verlet_Pos(double(*All_Pos)[3], double(*All_Vel)[3], double(*All_Acc)[3], double *dt);
	__global__ void Pos_period(double(*All_Pos)[3], double(*Box)[2], double(*Pt_ePos)[3]);
	__global__ void Acceleration_period(double(*All_Pos)[3], double(*All_Acc)[3], double *LJ_E, double *LJ_S, double(*Box)[2], double *cutoff, double(*Pt_ePos)[3], double *spr_k, double *Mass);
	__global__ void Verlet_Vel(double(*All_Vel)[3], double(*All_Acc_temp)[3], double(*All_Acc)[3], double *dt);
	__global__ void rescale_T2(double(*All_Vel)[3], double *Pt_argVelx, double *Pt_argVely, double *Pt_argVelz);
	__global__ void rescale_T4(double(*All_Vel)[3], double *Pt_T, double *T);

	Pars.Init();
	Pars.Initialization(All_type, All_Pos, All_Vel, All_Acc);
	timestep = 0;
	Pars.Dump(All_type, All_Pos, timestep);
	cudaMalloc((void**)&d_All_Pos, sizeof(All_Pos));
	cudaMalloc((void**)&d_All_Vel, sizeof(All_Vel));
	cudaMalloc((void**)&d_All_Acc, sizeof(All_Acc));
	cudaMalloc((void**)&d_All_Acc_temp, sizeof(All_Acc));
	cudaMalloc((void**)&d_dt, sizeof(double));
	cudaMalloc((void**)&d_Box, sizeof(Pars.Box));
	cudaMalloc((void**)&d_Pt_ePos, sizeof(Pars.Pt_ePos));
	cudaMalloc((void**)&d_Mass, sizeof(Pars.Mass));
	cudaMalloc((void**)&d_T, sizeof(Pars.T));
	cudaMalloc((void**)&d_LJ_E, sizeof(Pars.LJ_E));
	cudaMalloc((void**)&d_LJ_S, sizeof(Pars.LJ_S));
	cudaMalloc((void**)&d_cutoff, sizeof(double));
	cudaMalloc((void**)&d_spr_k, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVelx, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVely, sizeof(double));
	cudaMalloc((void**)&d_Pt_argVelz, sizeof(double));
	cudaMalloc((void**)&d_Pt_T, sizeof(double));
	cudaMemcpy(d_dt, &Pars.dt, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Box, Pars.Box, sizeof(Pars.Box), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Mass, Pars.Mass, sizeof(Pars.Mass), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, Pars.T, sizeof(Pars.T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_E, Pars.LJ_E, sizeof(Pars.LJ_E), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LJ_S, Pars.LJ_S, sizeof(Pars.LJ_S), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cutoff, &Pars.cutoff, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_spr_k, &Pars.spr_k, sizeof(double), cudaMemcpyHostToDevice);
	start = clock();
	while (Pars.state) {
		cudaMemcpy(d_All_Pos, All_Pos, sizeof(All_Pos), cudaMemcpyHostToDevice);
		cudaMemcpy(d_All_Vel, All_Vel, sizeof(All_Vel), cudaMemcpyHostToDevice);
		cudaMemcpy(d_All_Acc, All_Acc, sizeof(All_Acc), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Pt_ePos, Pars.Pt_ePos, sizeof(Pars.Pt_ePos), cudaMemcpyHostToDevice);
		Verlet_Pos << <1, Pt_N + Ar_N >> >(d_All_Pos, d_All_Vel, d_All_Acc, d_dt);
		cudaDeviceSynchronize();
		Pos_period << <1, Pt_N + Ar_N >> >(d_All_Pos, d_Box, d_Pt_ePos);
		cudaDeviceSynchronize();
		cudaMemcpy(All_Pos, d_All_Pos, sizeof(All_Pos), cudaMemcpyDeviceToHost);
		Pars.Box[2][0] = All_Pos[0][2];
		for (i = 0; i < Pt_N; i++) {
			if (All_Pos[i][2] < Pars.Box[2][0]) {
				Pars.Box[2][0] = All_Pos[i][2];
			}
		}
		cudaMemcpy(d_Box, Pars.Box, sizeof(Pars.Box), cudaMemcpyHostToDevice);
		for (i = 0; i < Pt_N + Ar_N; i++) {
			All_Acc_temp[i][0] = All_Acc[i][0];
			All_Acc_temp[i][1] = All_Acc[i][1];
			All_Acc_temp[i][2] = All_Acc[i][2];
		}
		cudaMemcpy(d_All_Acc_temp, All_Acc_temp, sizeof(All_Acc_temp), cudaMemcpyHostToDevice);
		Acceleration_period << <1, Pt_N + Ar_N >> >(d_All_Pos, d_All_Acc, d_LJ_E, d_LJ_S, d_Box, d_cutoff, d_Pt_ePos, d_spr_k, d_Mass);
		cudaDeviceSynchronize();
		Verlet_Vel << <1, Pt_N + Ar_N >> >(d_All_Vel, d_All_Acc_temp, d_All_Acc, d_dt);
		cudaDeviceSynchronize();
		cudaMemcpy(All_Vel, d_All_Vel, sizeof(All_Vel), cudaMemcpyDeviceToHost);
		Pars.rescale_T1(All_Vel);
		cudaMemcpy(d_Pt_argVelx, &Pars.Pt_argVelx, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Pt_argVely, &Pars.Pt_argVely, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Pt_argVelz, &Pars.Pt_argVelz, sizeof(double), cudaMemcpyHostToDevice);
		rescale_T2 << <1, Pt_N >> > (d_All_Vel, d_Pt_argVelx, d_Pt_argVely, d_Pt_argVelz);
		cudaDeviceSynchronize();
		cudaMemcpy(All_Vel, d_All_Vel, sizeof(All_Vel), cudaMemcpyDeviceToHost);
		Pars.rescale_T3(All_Vel);
		cudaMemcpy(d_Pt_T, &Pars.Pt_T, sizeof(double), cudaMemcpyHostToDevice);
		rescale_T4 << <1, Pt_N >> > (d_All_Vel, d_Pt_T, d_T);
		cudaDeviceSynchronize();
		cudaMemcpy(All_Pos, d_All_Pos, sizeof(All_Pos), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Vel, d_All_Vel, sizeof(All_Vel), cudaMemcpyDeviceToHost);
		cudaMemcpy(All_Acc, d_All_Acc, sizeof(All_Acc), cudaMemcpyDeviceToHost);
		cudaMemcpy(Pars.Pt_ePos, d_Pt_ePos, sizeof(Pars.Pt_ePos), cudaMemcpyDeviceToHost);
		timestep += 1;
		Pars.Exit(All_type, All_Pos, timestep);
		finish = clock();
		tperl = double(finish - start) / CLOCKS_PER_SEC / timestep;
		cout << timestep << " TimeSteps; ArgTime: " << tperl << " Seconds!\r";
	}
	cudaFree(d_All_Pos);
	cudaFree(d_All_Vel);
	cudaFree(d_All_Acc);
	cudaFree(d_All_Acc_temp);
	cudaFree(d_dt);
	cudaFree(d_Box);
	cudaFree(d_Pt_ePos);
	cudaFree(d_Mass);
	cudaFree(d_T);
	cudaFree(d_LJ_E);
	cudaFree(d_LJ_S);
	cudaFree(d_cutoff);
	cudaFree(d_spr_k);
	cudaFree(d_Pt_argVelx);
	cudaFree(d_Pt_argVely);
	cudaFree(d_Pt_argVelz);
	cudaFree(d_Pt_T);
	cout << "\n";
	system("pause");
	return 0;
}