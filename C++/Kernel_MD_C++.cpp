#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include<algorithm>
using namespace std;
const int Pt_I = 6, Pt_J = 6, Pt_K = 3;

class Parameters {
public:
	int Tt, Pt_N, Pt_type[4 * Pt_I*Pt_J*Pt_K], Ar_N, Ar_type[1], dumpstep;
	double PI, Mass[2], LJ_E[3], LJ_S[3], fcc_lattice, kB, T[2], mp_V[2];
	double nd_Mass, nd_Energy, nd_Length, nd_Velocity, nd_Time, nd_Acceleration;
	double cutoff, d, spr_k, dt;
	double Box[3][3], Pt_ePos[4 * Pt_I*Pt_J*Pt_K][3];
	bool state;
	void Init();
	int Initialization(double Pt_Pos[][3], double Pt_Vel[][3], double Pt_Acc[][3], double Ar_Pos[][3], double Ar_Vel[][3], double Ar_Acc[][3]);
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
	Mass[0] = 39.95 / 6.02 * 1E-26;//单位kg
	Mass[1] = 195.08 / 6.02 * 1E-26;
	LJ_E[0] = 1.654E-21;//单位J
	LJ_E[1] = 5.207E-20;
	LJ_E[2] = 1.093E-21;
	LJ_S[0] = 3.40 * 1E-10;//单位m
	LJ_S[1] = 2.47 * 1E-10;
	LJ_S[2] = 2.94 * 1E-10;
	fcc_lattice = 3.93E-10;
	kB = 1.38E-23;
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
	Tt = 3500000;
	//盒子参数
	Pt_N = 0;
	Ar_N = 0;
	//状态参数
	state = true;
	dumpstep = 100;

	cout << "Parameters Initialized!\n";
}

/******************************************************************************/
int Parameters::Initialization(double Pt_Pos[][3], double Pt_Vel[][3], double Pt_Acc[][3], double Ar_Pos[][3], double Ar_Vel[][3], double Ar_Acc[][3]) {
	int timestep, count, i, j, k, dim;
	double R1, R2, Pt_V2, Pt_T, sumV[3];

	timestep = 0;
	//初始化Pt的初始位置和速度
	count = 0;
	srand((unsigned)time(NULL));
	for (i = 0;i < 2 * Pt_I;i++) {
		for (j = 0;j < 2 * Pt_J;j++) {
			for (k = 0;k < 2 * Pt_K;k++) {
				if (i / 2. + j / 2. + k / 2. == int(i / 2. + j / 2. + k / 2.)) {
					count++;
					Pt_type[count - 1] = 1;
					Pt_Pos[count - 1][0] = i / 2.*fcc_lattice;
					Pt_Pos[count - 1][1] = j / 2.*fcc_lattice;
					Pt_Pos[count - 1][2] = (k / 2. - 2.5)*fcc_lattice;
					for (dim = 0;dim < 3;dim++) {
						R1 = random();
						R2 = random();
						Pt_Vel[count - 1][dim] = mp_V[1] / sqrt(3)*sqrt(-2 * log(R1))*cos(2 * PI*R2);//��˹�ֲ���ƽ��ֵΪ0������=�������ٶ���������
					}
				}
			}
		}
	}
	Pt_N = count;
	Box[0][0] = Pt_Pos[0][0];
	Box[0][1] = Pt_Pos[0][0];
	Box[1][0] = Pt_Pos[0][1];
	Box[1][1] = Pt_Pos[0][1];
	Box[2][0] = Pt_Pos[0][2];
	Box[2][1] = Pt_Pos[0][2];
	for (i = 0;i < Pt_N;i++) {
		for (j = 0;j < 3;j++) {
			if (Pt_Pos[i][j] < Box[j][0]) {
				Box[j][0] = Pt_Pos[i][j];
			}
			if (Pt_Pos[i][j] > Box[j][1]) {
				Box[j][1] = Pt_Pos[i][j];
			}
		}
	}
	Box[0][1] += fcc_lattice / 2;
	Box[1][1] += fcc_lattice / 2;
	Box[2][1] = d;
	Box[0][2] = Box[0][1] - Box[0][0];
	Box[1][2] = Box[1][1] - Box[1][0];
	Box[2][2] = Box[2][1] - Box[2][0];
	cout << "Zone[X]: " << Box[0][0] << ", " << Box[0][1] << "\n";
	cout << "Zone[Y]: " << Box[1][0] << ", " << Box[1][1] << "\n";
	cout << "Zone[Z]: " << Box[2][0] << ", " << Box[2][1] << "\n";
	//初始化Ar的初始位置和速度
	Ar_N = 1;
	Ar_type[0] = 0;
	Ar_Pos[0][0] = Box[0][0] + Box[0][2] * (rand() / double(RAND_MAX));
	Ar_Pos[0][1] = Box[1][0] + Box[1][2] * (rand() / double(RAND_MAX));
	Ar_Pos[0][2] = Box[2][1];
	R1 = random();
	R2 = random();
	Ar_Vel[0][0] = mp_V[0] * sqrt(-log(R1))*cos(2 * PI*R2);//Maxwell�ֲ�
	R1 = random();
	R2 = random();
	Ar_Vel[0][1] = mp_V[0] * sqrt(-log(R1))*sin(2 * PI*R2);
	R1 = random();
	Ar_Vel[0][2] = -mp_V[0] * sqrt(-log(R1));
	//初始信息
	cout << "Created " << Pt_N << " Pt\n";
	cout << "Created " << Ar_N << " Ar\n";
	sumV[0] = 0.;
	sumV[1] = 0.;
	sumV[2] = 0.;
	Pt_V2 = 0.;
	for (i = 0;i < Pt_N;i++) {
		sumV[0] += Pt_Vel[i][0];
		sumV[1] += Pt_Vel[i][1];
		sumV[2] += Pt_Vel[i][2];
		Pt_V2 += pow(Pt_Vel[i][0], 2) + pow(Pt_Vel[i][1], 2) + pow(Pt_Vel[i][2], 2);
	}
	Pt_T = Pt_V2 * pow(nd_Velocity, 2) * Mass[1] * nd_Mass / (3 * Pt_N*kB);
	cout << "Pt argV in X: " << sumV[0] / Pt_N << "\n";
	cout << "Pt argV in Y: " << sumV[1] / Pt_N << "\n";
	cout << "Pt argV in Z: " << sumV[2] / Pt_N << "\n";
	cout << "Pt temp: " << Pt_T << " K\n";
	cout << "Ar V: " << Ar_Vel[0][0] << ", " << Ar_Vel[0][1] << ", " << Ar_Vel[0][2] << "\n";
	cout << "*******Model Initialization Done!*******\n";
	//首次控温
	rescale_T(Pt_Vel);
	for (i = 0;i < Pt_N;i++) {
		for (j = 0;j < 3;j++) {
			Pt_ePos[i][j] = Pt_Pos[i][j];
		}
	}
	//首次位置周期
	Pos_period(Ar_Pos, Pt_Pos);
	//首次加速度周期
	Acceleration_period(Ar_Pos, Pt_Pos, Ar_Acc, Pt_Acc);

	return timestep;
}

/******************************************************************************/
void Parameters::Pos_period(double Ar_Pos[][3], double Pt_Pos[][3]) {
	int axis, i;
	//X,Y方向周期
	for (axis = 0;axis < 2;axis++) {
		//Pt
		for (i = 0;i < Pt_N;i++) {
			if (Pt_Pos[i][axis] < Box[axis][0]) {
				Pt_Pos[i][axis] += Box[axis][2];
				Pt_ePos[i][axis] += Box[axis][2];
			}
			else if (Pt_Pos[i][axis] >= Box[axis][1]) {
				Pt_Pos[i][axis] -= Box[axis][2];
				Pt_ePos[i][axis] -= Box[axis][2];
			}
		}
		//Ar
		if (Ar_Pos[0][axis] < Box[axis][0]) {
			Ar_Pos[0][axis] += Box[axis][2];
		}
		else if (Ar_Pos[0][axis] >= Box[axis][1]) {
			Ar_Pos[0][axis] -= Box[axis][2];
		}
	}
	//Z方向下边界更新
	Box[2][0] = Pt_Pos[0][2];
	for (i = 0;i < Pt_N;i++) {
		if (Pt_Pos[i][2] < Box[2][0]) {
			Box[2][0] = Pt_Pos[i][2];
		}
	}
	Box[2][2] = Box[2][1] - Box[2][0];
}

/******************************************************************************/
void Parameters::Acceleration_period(double Ar_Pos[][3], double Pt_Pos[][3], double Ar_Acc[][3], double Pt_Acc[][3]) {
	double All_Pos[4 * Pt_I*Pt_J*Pt_K + 1][3], Pt_F[4 * Pt_I*Pt_J*Pt_K][3], Ar_F[3], Atom_F[3], Pair[3], Spring_Dis[4 * Pt_I*Pt_J*Pt_K][3], Spring_F[4 * Pt_I*Pt_J*Pt_K][3];
	int All_type[4 * Pt_I*Pt_J*Pt_K + 1];
	int All_N, LJ_pair, i, j, axis;
	double Epair, Spair, Dispair, Fpair;

	for (j = 0;j < 3;j++) {
		for (i = 0;i < Pt_N;i++) {
			All_Pos[i][j] = Pt_Pos[i][j];
			All_type[i] = Pt_type[i];
		}
		All_Pos[Pt_N][j] = Ar_Pos[0][j];
		All_type[Pt_N] = Ar_type[0];
	}
	All_N = Pt_N + Ar_N;
	for (i = 0;i < All_N;i++) {
		Atom_F[0] = 0.;
		Atom_F[1] = 0.;
		Atom_F[2] = 0.;
		for (j = 0;j < All_N;j++) {
			if (All_type[i] == 1 && All_type[j] == 1) {
				LJ_pair = 1;
			}
			else if (All_type[i] == 0 && All_type[j] == 0) {
				LJ_pair = 0;
			}
			else {
				LJ_pair = 2;
			}
			Epair = LJ_E[LJ_pair];
			Spair = LJ_S[LJ_pair];
			//周期相对位置
			for (axis = 0;axis < 3;axis++) {
				Pair[axis] = All_Pos[i][axis] - All_Pos[j][axis];
				if (axis != 2 && abs(Pair[axis]) >= Box[axis][2] - cutoff) {
					Pair[axis] = Pair[axis] - Box[axis][2] * Pair[axis] / abs(Pair[axis]);
				}
			}
			//周期距离
			Dispair = sqrt(pow(Pair[0], 2.) + pow(Pair[1], 2.) + pow(Pair[2], 2.));
			if (Dispair > 0 && Dispair <= cutoff) {
				Fpair = 48 * Epair * (pow(Spair, 12.) / pow(Dispair, 13.) - 0.5 * pow(Spair, 6.) / pow(Dispair, 7.));
				for (axis = 0;axis < 3;axis++) {
					Atom_F[axis] += Pair[axis] * Fpair / Dispair;
				}
			}
		}
		//Pt弹性恢复力
		if (All_type[i] == 1) {
			for (axis = 0;axis < 3;axis++) {
				Spring_Dis[i][axis] = Pt_Pos[i][axis] - Pt_ePos[i][axis];
				Spring_F[i][axis] = -spr_k * Spring_Dis[i][axis];
				Pt_F[i][axis] = Atom_F[axis] + Spring_F[i][axis];
				Pt_Acc[i][axis] = Pt_F[i][axis] / Mass[1];
			}
		}
		else {
			for (axis = 0;axis < 3;axis++) {
				Ar_F[axis] = Atom_F[axis];
				Ar_Acc[0][axis] = Ar_F[axis] / Mass[0];
			}
		}
	}
}

/******************************************************************************/
void Parameters::Verlet(double Ar_Pos[][3], double Ar_Vel[][3], double Ar_Acc[][3], double Pt_Pos[][3], double Pt_Vel[][3], double Pt_Acc[][3]) {
	double Ar_Acc_n[1][3], Pt_Acc_n[4 * Pt_I*Pt_J*Pt_K][3];
	int axis, i;

	for (axis = 0;axis < 3;axis++) {
		for (i = 0;i < Pt_N;i++) {
			Pt_Pos[i][axis] += Pt_Vel[i][axis] * dt + 0.5*Pt_Acc[i][axis] * pow(dt, 2.);
		}
		Ar_Pos[0][axis] += Ar_Vel[0][axis] * dt + 0.5*Ar_Acc[0][axis] * pow(dt, 2.);
	}
	Pos_period(Ar_Pos, Pt_Pos);
	Acceleration_period(Ar_Pos, Pt_Pos, Ar_Acc_n, Pt_Acc_n);
	for (axis = 0;axis < 3;axis++) {
		for (i = 0;i < Pt_N;i++) {
			Pt_Vel[i][axis] += 0.5*(Pt_Acc[i][axis] + Pt_Acc_n[i][axis])*dt;
			Pt_Acc[i][axis] = Pt_Acc_n[i][axis];
		}
		Ar_Vel[0][axis] += 0.5*(Ar_Acc[0][axis] + Ar_Acc_n[0][axis])*dt;
		Ar_Acc[0][axis] = Ar_Acc_n[0][axis];
	}
}

/******************************************************************************/
void Parameters::rescale_T(double Pt_Vel[][3]) {
	double Pt_argVel[3], Pt_V2, Pt_T;
	int axis, i;

	for (axis = 0;axis < 3;axis++) {
		Pt_argVel[axis] = 0.;
		for (i = 0;i < Pt_N;i++) {
			Pt_argVel[axis] += Pt_Vel[i][axis];
		}
		Pt_argVel[axis] /= Pt_N;
		for (i = 0;i < Pt_N;i++) {
			Pt_Vel[i][axis] -= Pt_argVel[axis];
		}
	}
	Pt_V2 = 0.;
	for (i = 0;i < Pt_N;i++) {
		Pt_V2 += pow(Pt_Vel[i][0], 2) + pow(Pt_Vel[i][1], 2) + pow(Pt_Vel[i][2], 2);
	}
	Pt_T = Pt_V2 * pow(nd_Velocity, 2) * Mass[1] * nd_Mass / (3 * Pt_N*kB);
	for (i = 0;i < Pt_N;i++) {
		for (axis = 0;axis < 3;axis++) {
			Pt_Vel[i][axis] *= sqrt(T[1] / Pt_T);
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
void Parameters::Exit(double Pt_Pos[][3], double Ar_Pos[][3], int timestep) {

	if (Ar_Pos[0][2] > d || timestep >= Tt) {
		state = false;
		Dump(Pt_Pos, Ar_Pos, timestep);
	}
	else {
		Dump(Pt_Pos, Ar_Pos, timestep, dumpstep);
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
