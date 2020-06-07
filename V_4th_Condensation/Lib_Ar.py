'''
Name:              Lib_Ar.py
Description:       Create a library of incident atoms and a wall model
Auther:            KhalilWong
License:           MIT License
Copyright:         Copyright (C) 2018, by KhalilWong
Version:           3.0.0
Date:              2018/10/27
Namespace:         https://github.com/KhalilWong
DownloadURL:       https://github.com/KhalilWong/...
UpdateURL:         https://github.com/KhalilWong/...
'''
################################################################################
import time
import random
import math
import matplotlib.pyplot as mpl
import numpy as np
import Time_Advancement as TA
import Create_Gif as CG
from numba import cuda
import matplotlib.ticker as ticker
################################################################################
Lib_N = 200000
Pt_I = 6
Pt_J = 6
Pt_K = 3
Pt_N = 4 * Pt_I * Pt_J * Pt_K
################################################################################
class Dim_Parameters():
    #
    def __init__(self, Mass, Energy, Length):
        self.Mass = Mass
        self.Energy = Energy
        self.Length = Length
        self.Velocity = math.sqrt(self.Energy / self.Mass)
        self.Time = self.Length / self.Velocity
        self.Acceleration = self.Energy / (self.Mass * self.Length)

################################################################################
class MD_Parameters():
    #
    def __init__(self):
        self.pi = 3.14159265
        self.kB = 1.38E-23
        self.fcc_lattice = 3.93E-10
        self.cutoff = 10 * 1E-10
        self.Spr_K = 5000
        self.BoxXLow = 0
        self.BoxXHigh = Pt_I * self.fcc_lattice
        self.BoxYLow = 0
        self.BoxYHigh = Pt_J * self.fcc_lattice
        #self.BoxZLow = - (Pt_K-0.5) * self.fcc_lattice
        self.BoxZLow = - Pt_K * self.fcc_lattice
        self.BoxZHigh = 5.0
        self.Period = 10000
        self.Period_WallT = []
        self.TimeStep = 0
        self.dt = 0.001
        self.DumpStep = 100
        self.State = True
    #
    def Non_Dim(self, Dim):
        self.fcc_lattice /= Dim.Length
        self.cutoff /= Dim.Length
        self.BoxXLow /= Dim.Length
        self.BoxXHigh /= Dim.Length
        self.BoxYLow /= Dim.Length
        self.BoxYHigh /= Dim.Length
        self.BoxZLow /= Dim.Length
    #
    def MD_Dump(self, Atoms):
        if self.TimeStep >= self.Period:
            self.State = False
            self.DumpStep = 1
            with open('Pt_Wall.data', 'w') as Init:
                print('ITEM: TIMESTEP', file = Init)
                print(0, file = Init)
                print('ITEM: NUMBER OF ATOMS', file = Init)
                print(Pt_N, file = Init)
                print('ITEM: BOX BOUNDS pp pp ff', file = Init)
                print(self.BoxXLow, self.BoxXHigh, file = Init)
                print(self.BoxYLow, self.BoxYHigh, file = Init)
                print(self.BoxZLow, self.BoxZHigh, file = Init)
                print('ITEM: ATOMS id type elas x y z vx vy vz ax ay az', file = Init)
                for i in range(Pt_N):
                    print(Atoms[i].ID, Atoms[i].Type, Atoms[i].Elasticity, Atoms[i].x, Atoms[i].y, Atoms[i].z, Atoms[i].vx, Atoms[i].vy, Atoms[i].vz, Atoms[i].ax, Atoms[i].ay, Atoms[i].az, file = Init)
            with open('Pt_Wall.balance', 'w') as Bal:
                print('ITEM: NUMBER OF ATOMS', file = Bal)
                print(Pt_N, file = Bal)
                print('ITEM: ATOMS id Bx By Bz', file = Bal)
                for i in range(Pt_N):
                    print(Atoms[i].ID, Atoms[i].Bx, Atoms[i].By, Atoms[i].Bz, file = Bal)
        if self.TimeStep % self.DumpStep == 0:
            with open('Pt_Wall.relaxation', 'a') as R_MD:
                print('ITEM: TIMESTEP', file = R_MD)
                print(self.TimeStep, file = R_MD)
                print('ITEM: NUMBER OF ATOMS', file = R_MD)
                print(Pt_N, file = R_MD)
                print('ITEM: BOX BOUNDS pp pp ff', file = R_MD)
                print(self.BoxXLow, self.BoxXHigh, file = R_MD)
                print(self.BoxYLow, self.BoxYHigh, file = R_MD)
                print(self.BoxZLow, self.BoxZHigh, file = R_MD)
                print('ITEM: ATOMS id x y z vx vy vz ax ay az', file = R_MD)
                for i in range(Pt_N):
                    print(Atoms[i].ID, Atoms[i].x, Atoms[i].y, Atoms[i].z, Atoms[i].vx, Atoms[i].vy, Atoms[i].vz, Atoms[i].ax, Atoms[i].ay, Atoms[i].az, file = R_MD)

################################################################################
class Atom():
    #
    def __init__(self):
        self.ID = 0
        self.Type = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.Bx = 0.0
        self.By = 0.0
        self.Bz = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.Elasticity = False

################################################################################
class Atom_Parameters():
    #
    def __init__(self, Mass, Epsilon, Sigma):
        self.Mass = Mass
        self.Epsilon = Epsilon
        self.Sigma = Sigma
        self.T = 0.0
        self.mpVelocity = 0.0
    #
    def Set_T(self, kB, T, Axis = 3):
        self.T = T
        self.mpVelocity = math.sqrt(Axis * kB * self.T / self.Mass)
    #
    def Non_Dim(self, Dim):
        self.Mass /= Dim.Mass
        self.Epsilon /= Dim.Energy
        self.Sigma /= Dim.Length
        self.mpVelocity /= Dim.Velocity

################################################################################
def Rescale_T(Atoms, Atom_Pars, MD_Pars, Dim, N):
    #只需要热运动速度
    ArgVelX = 0.0
    ArgVelY = 0.0
    ArgVelZ = 0.0
    for i in range(N):
        ArgVelX += Atoms[i].vx
        ArgVelY += Atoms[i].vy
        ArgVelZ += Atoms[i].vz
    ArgVelX /= N
    ArgVelY /= N
    ArgVelZ /= N
    for i in range(N):
        Atoms[i].vx -= ArgVelX
        Atoms[i].vy -= ArgVelY
        Atoms[i].vz -= ArgVelZ
    CurrentT = 0.0
    for i in range(N):
        CurrentT += Atoms[i].vx ** 2 + Atoms[i].vy ** 2 + Atoms[i].vz ** 2
    CurrentT = CurrentT * Dim.Velocity ** 2 * Atom_Pars.Mass * Dim.Mass / (3 * N * MD_Pars.kB)
    MD_Pars.Period_WallT.append(CurrentT)
    for i in range(N):
        Atoms[i].vx *= math.sqrt(Atom_Pars.T / CurrentT)
        Atoms[i].vy *= math.sqrt(Atom_Pars.T / CurrentT)
        Atoms[i].vz *= math.sqrt(Atom_Pars.T / CurrentT)

################################################################################
def Acceleration(Atoms, Atom_Pars, MD_Pars, N):
    for i in range(N):
        Atom_Fx = 0.0
        Atom_Fy = 0.0
        Atom_Fz = 0.0
        for j in range(N):
            #周期相对位置
            Pairx = Atoms[i].x - Atoms[j].x
            Pairy = Atoms[i].y - Atoms[j].y
            Pairz = Atoms[i].z - Atoms[j].z
            if abs(Pairx) >= MD_Pars.BoxXHigh - MD_Pars.BoxXLow - MD_Pars.cutoff:
                Pairx = Pairx - (MD_Pars.BoxXHigh - MD_Pars.BoxXLow) * Pairx / abs(Pairx)
            if abs(Pairy) >= MD_Pars.BoxYHigh - MD_Pars.BoxYLow - MD_Pars.cutoff:
                Pairy = Pairy - (MD_Pars.BoxYHigh - MD_Pars.BoxYLow) * Pairy / abs(Pairy)
            #周期距离
            Dispair = math.sqrt(Pairx ** 2 + Pairy ** 2 + Pairz ** 2)
            if 0 < Dispair <= MD_Pars.cutoff:
                Fpair = 48 * Atom_Pars.Epsilon * (Atom_Pars.Sigma ** 12 / Dispair ** 13 - 0.5 * Atom_Pars.Sigma ** 6 / Dispair ** 7)
                Atom_Fx += Pairx * Fpair / Dispair
                Atom_Fy += Pairy * Fpair / Dispair
                Atom_Fz += Pairz * Fpair / Dispair
        #Pt弹性恢复力
        Spring_Disx = Atoms[i].x - Atoms[i].Bx
        Spring_Disy = Atoms[i].y - Atoms[i].By
        Spring_Disz = Atoms[i].z - Atoms[i].Bz
        Spring_Fx = - MD_Pars.Spr_K * Spring_Disx
        Spring_Fy = - MD_Pars.Spr_K * Spring_Disy
        Spring_Fz = - MD_Pars.Spr_K * Spring_Disz
        Atom_Fx += Spring_Fx
        Atom_Fy += Spring_Fy
        Atom_Fy += Spring_Fz
        Atoms[i].ax = Atom_Fx / Atom_Pars.Mass
        Atoms[i].ay = Atom_Fy / Atom_Pars.Mass
        Atoms[i].az = Atom_Fz / Atom_Pars.Mass

################################################################################
def Lib_Dump(Atoms):
    with open('Lib_Ar.data', 'w') as Lib:
        print('ITEM: LIB_N', file = Lib)
        print(Lib_N, file = Lib)
        print('ITEM: ATOMS id type elas x y z vx vy vz ax ay az', file = Lib)
        for i in range(Lib_N):
            print(Atoms[i].ID, Atoms[i].Type, Atoms[i].Elasticity, Atoms[i].x, Atoms[i].y, Atoms[i].z, Atoms[i].vx, Atoms[i].vy, Atoms[i].vz, Atoms[i].ax, Atoms[i].ay, Atoms[i].az, file = Lib)

################################################################################
def Dis_Vel(Name, Atoms, N, mpVelocity, Axis, XL, XH, YL, YH):
    pi = 3.14159265
    AccN = 30
    Temp = [int(round((math.sqrt(Atoms[i].vx ** 2 + Atoms[i].vy ** 2 + Atoms[i].vz ** 2)) * AccN)) for i in range(N)]
    Max = max(Temp)
    Min = min(Temp)
    Delta = Min
    X = []
    Y = []
    while Min <= Delta <= Max:
        X.append(Delta / AccN)
        Y.append(Temp.count(Delta) * AccN / N)
        Delta += 1
    MX = [x / AccN / 10 for x in range(Min * 10, (Max + 1) * 10)]
    MY = [4 * pi * (Axis / (2 * pi * mpVelocity ** 2)) ** (3 / 2) * math.exp(- Axis * MXi ** 2 / (2 * mpVelocity ** 2)) * MXi ** 2 for MXi in MX]
    #
    Temp = [int(round(Atoms[i].vx / mpVelocity * AccN)) for i in range(N)]
    Max = max(Temp)
    Min = min(Temp)
    Delta = Min
    X1 = []
    Y1 = []
    while Min <= Delta <= Max:
        X1.append(Delta / AccN)
        Y1.append(Temp.count(Delta) * AccN / N)
        Delta += 1
    MX1 = [x / AccN / 10 for x in range(Min * 10, (Max + 1) * 10)]
    #MY1=[math.sqrt(Axis/(2*pi*mpVelocity**2))*math.exp(-Axis*MX1i**2/(2*mpVelocity**2)) for MX1i in MX1]
    MY1 = [math.sqrt(Axis / (2 * pi)) * math.exp(- Axis * MX1i ** 2 / (2)) for MX1i in MX1]
    #
    Temp = [int(round(Atoms[i].vy / mpVelocity * AccN)) for i in range(N)]
    Max = max(Temp)
    Min = min(Temp)
    Delta = Min
    X2 = []
    Y2 = []
    while Min <= Delta <= Max:
        X2.append(Delta / AccN)
        Y2.append(Temp.count(Delta) * AccN / N)
        Delta += 1
    MX2 = [x / AccN / 10 for x in range(Min * 10, (Max + 1) * 10)]
    #MY2=[math.sqrt(Axis/(2*pi*mpVelocity**2))*math.exp(-Axis*MX2i**2/(2*mpVelocity**2)) for MX2i in MX2]
    MY2 = [math.sqrt(Axis / (2 * pi)) * math.exp(- Axis * MX2i ** 2 / (2)) for MX2i in MX2]
    #
    Temp = [int(round(Atoms[i].vz / mpVelocity * AccN)) for i in range(N)]
    Max = max(Temp)
    Min = min(Temp)
    Delta = Min
    X3 = []
    Y3 = []
    while Min <= Delta <= Max:
        X3.append(Delta / AccN)
        Y3.append(Temp.count(Delta) * AccN / N)
        Delta += 1
    MX3 = [x / AccN / 10 for x in range(Min * 10, (Max + 1) * 10)]
    #MY3=[abs(Axis*MX3i/mpVelocity**2*math.exp(-Axis*MX3i**2/(2*mpVelocity**2))) for MX3i in MX3]
    MY3 = [abs(Axis * MX3i * math.exp(- Axis * MX3i ** 2 / (2))) for MX3i in MX3]
    #
    fig = mpl.figure()
    lab = 'Velocity Distribution of ' + str(N) + ' ' + Name + ' Atoms'
    mpl.plot(X, Y, 'ro', markersize = 4, label = lab)
    mpl.plot(MX, MY, 'b', markersize = 1, label = 'Maxwell-Boltzmann distribution')
    #rmsx = [math.sqrt(3 * mpVelocity ** 2 / Axis) for x in range(100)]
    #rmsy = [y / 100 for y in range(100)]
    #mpl.plot(rmsx,rmsy,'g--',markersize=1,label='Root Mean Square Speed')
    #mpx = [math.sqrt(2 * mpVelocity ** 2 / Axis) for x in range(100)]
    #mpy = [y / 100 for y in range(100)]
    #mpl.plot(mpx,mpy,'g',markersize=1,label='Most Probable Speed')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$v$')
    mpl.ylabel('$D$')
    mpl.axis([XL, XH, YL, YH])
    mpl.savefig(Name + '_D_v.png', dpi = 600)
    #
    #fig2=mpl.figure()
    #lab='X-Velocity Distribution of '+str(N)+' '+Name+' Atoms'
    #mpl.plot(X1,Y1,'ro',markersize=4,label=lab)
    #mpl.plot(MX1,MY1,'b',markersize=1,label='Maxwellian')
    #mpl.legend(loc='upper right')
    #mpl.xlabel('$vx$')
    #mpl.ylabel('$D$')
    #mpl.axis([-3.0,3.0,0.0,0.8])
    #mpl.savefig(Name+'_D_vx.png',dpi=600)
    #
    #fig3=mpl.figure()
    #lab='Y-Velocity Distribution of '+str(N)+' '+Name+' Atoms'
    #mpl.plot(X2,Y2,'ro',markersize=4,label=lab)
    #mpl.plot(MX2,MY2,'b',markersize=1,label='Maxwellian')
    #mpl.legend(loc='upper right')
    #mpl.xlabel('$vy$')
    #mpl.ylabel('$D$')
    #mpl.axis([-3.0,3.0,0.0,0.8])
    #mpl.savefig(Name+'_D_vy.png',dpi=600)
    #
    fig4, ax4 = mpl.subplots()
    #ax4.set_title('b',loc='left')
    ax4.text(0.95, 0.95, '(b)', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 12)
    ax4.plot(X3, Y3, 'go', markersize = 3, label = '$w_i$')
    ax4.plot(MX3, MY3, 'm', markersize = 3, label = 'Maxwellian')
    ax4.legend(loc = 'upper left', fontsize = 'medium', frameon = False)
    ax4.set_xlabel('$w$')
    ax4.set_ylabel('$f$')
    ax4.set_xlim(-3.0, 0.0)
    ax4.set_ylim(0.0, 1.0)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax4.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax4.tick_params(axis = 'both', which = 'major', direction = 'in', length = 5, width = 2)
    ax4.tick_params(axis = 'both', which = 'minor', direction = 'in', length = 3, width = 1)
    ax4.spines['left'].set_linewidth(2)
    ax4.spines['bottom'].set_linewidth(2)
    #
    ax5 = ax4.twinx()
    ax5.set_ylim(0.0, 1.0)
    ax5.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax5.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax5.yaxis.set_major_formatter(ticker.NullFormatter())
    ax5.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax5.tick_params(axis = 'y', which = 'major', direction = 'in', length = 5, width = 2)
    ax5.tick_params(axis = 'y', which = 'minor', direction = 'in', length = 3, width = 1)
    ax5.spines['right'].set_linewidth(2)
    #
    ax6 = ax4.twiny()
    ax6.set_xlim(-3.0, 0.0)
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax6.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax6.xaxis.set_major_formatter(ticker.NullFormatter())
    ax6.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax6.tick_params(axis = 'x', which = 'major', direction = 'in', length = 5, width = 2)
    ax6.tick_params(axis = 'x', which = 'minor', direction = 'in', length = 3, width = 1)
    ax6.spines['top'].set_linewidth(2)
    #
    mpl.savefig(Name + '_f_vn.eps', dpi = 600)
    #
    fig5, ax1 = mpl.subplots()
    #ax1.set_title('a',loc='left')
    ax1.text(0.05, 0.95, '(a)', horizontalalignment = 'center', verticalalignment = 'center', transform = ax1.transAxes, fontsize = 12)
    ax1.set_xlabel('$u$')
    ax1.set_xlim(-3.0, 3.0)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.set_ylabel('$f$')
    ax1.set_ylim(0.0, 0.7)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    l1 = ax1.plot(X1, Y1, 'ro', markersize = 3, label = '$u_i$')
    l3 = ax1.plot(MX1, MY1, 'm', markersize = 3, label = 'Maxwellian')
    ax1.tick_params(axis = 'both', which = 'major', direction = 'in', length = 5, width = 2)
    ax1.tick_params(axis = 'both', which = 'minor', direction = 'in', length = 3, width = 1)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    #
    ax2 = ax1.twiny()
    ax2.set_xlabel('$v$')
    ax2.set_xlim(-3.0, 3.0)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    l2 = ax2.plot(X2, Y2, 'bv', markersize = 3, label = '$v_i$')
    ax2.tick_params(axis = 'x', which = 'major', direction = 'in', length = 5, width = 2)
    ax2.tick_params(axis = 'x', which = 'minor', direction = 'in', length = 3, width = 1)
    ax2.spines['top'].set_linewidth(2)
    #
    ax3 = ax1.twinx()
    ax3.set_ylim(0.0, 0.7)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax3.yaxis.set_major_formatter(ticker.NullFormatter())
    ax3.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.tick_params(axis = 'y', which = 'major', direction = 'in', length = 5, width = 2)
    ax3.tick_params(axis = 'y', which = 'minor', direction = 'in', length = 3, width = 1)
    ax3.spines['right'].set_linewidth(2)
    #
    ls = l1 + l2 + l3
    legs = [l.get_label() for l in ls]
    ax2.legend(ls, legs, loc = 'upper right', fontsize = 'medium', frameon = False)#medium
    #
    mpl.savefig(Name + '_f_vt.eps', dpi = 600)
    #
    #mpl.show()
    mpl.close()

################################################################################
def Dis_Pos(Name, Atoms, MD_Pars, N):
    AccN = 30
    Temp = [int(round(Atoms[i].x * AccN)) for i in range(N)]
    Max = max(Temp)
    Min = min(Temp)
    Delta = Min
    X = []
    Y = []
    while Min <= Delta <= Max:
        X.append(Delta / AccN)
        Y.append(Temp.count(Delta) * AccN / N)
        Delta += 1
    #
    Temp = [int(round(Atoms[i].y * AccN)) for i in range(N)]
    Max = max(Temp)
    Min = min(Temp)
    Delta = Min
    X1 = []
    Y1 = []
    while Min <= Delta <= Max:
        X1.append(Delta / AccN)
        Y1.append(Temp.count(Delta) * AccN / N)
        Delta += 1
    #
    fig1 = mpl.figure()
    lab = 'X-Position Distribution of ' + str(N) + ' ' + Name + ' Atoms'
    mpl.plot(X, Y, 'ro', markersize = 4, label = lab)
    lowx = [MD_Pars.BoxXLow for i in range(100)]
    lowy = [y / 100 for y in range(100)]
    mpl.plot(lowx, lowy, 'r--', markersize = 1, label = 'XLow')
    highx = [MD_Pars.BoxXHigh for i in range(100)]
    highy = [y / 100 for y in range(100)]
    mpl.plot(highx, highy, 'g--', markersize = 1, label = 'XHigh')
    unix = [x / 100 for x in range(int(MD_Pars.BoxXLow * 100), int(MD_Pars.BoxXHigh * 100))]
    uniy = [1 / (MD_Pars.BoxXHigh - MD_Pars.BoxXLow) for i in range(int(MD_Pars.BoxXLow * 100), int(MD_Pars.BoxXHigh * 100))]
    mpl.plot(unix, uniy, 'b', markersize = 1, label = 'Uniform')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$x$')
    mpl.ylabel('$D$')
    mpl.axis([-1.0, 10.0, 0.0, 1.0])
    mpl.savefig(Name + '_D_x.png', dpi = 600)
    #
    fig2 = mpl.figure()
    lab = 'Y-Position Distribution of ' + str(N) + ' ' + Name + ' Atoms'
    mpl.plot(X1, Y1, 'ro', markersize = 4, label = lab)
    lowx = [MD_Pars.BoxYLow for i in range(100)]
    lowy = [y / 100 for y in range(100)]
    mpl.plot(lowx, lowy, 'r--', markersize = 1, label = 'YLow')
    highx = [MD_Pars.BoxYHigh for i in range(100)]
    highy = [y / 100 for y in range(100)]
    mpl.plot(highx, highy, 'g--', markersize = 1, label = 'YHigh')
    unix = [x / 100 for x in range(int(MD_Pars.BoxYLow * 100), int(MD_Pars.BoxYHigh * 100))]
    uniy = [1 / (MD_Pars.BoxYHigh - MD_Pars.BoxYLow) for i in range(int(MD_Pars.BoxYLow * 100), int(MD_Pars.BoxYHigh * 100))]
    mpl.plot(unix, uniy, 'b', markersize = 1, label = 'Uniform')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$y$')
    mpl.ylabel('$D$')
    mpl.axis([-1.0, 10.0, 0.0, 1.0])
    mpl.savefig(Name + '_D_y.png', dpi = 600)
    #
    #mpl.show()
    mpl.close()
################################################################################
if __name__ == '__main__':
    #
    Dim = Dim_Parameters(195.08 / 6.02 * 1E-26, 5.207E-20, 2.47E-10)
    MD_Pars = MD_Parameters()
    MD_Pars.Non_Dim(Dim)
    Pt_Pars = Atom_Parameters(195.08 / 6.02 * 1E-26, 5.207E-20, 2.47E-10)
    Pt_Pars.Set_T(MD_Pars.kB, 300.0)
    Pt_Pars.Non_Dim(Dim)
    Ar_Pars = Atom_Parameters(39.95 / 6.02 * 1E-26, 1.654E-21, 3.40E-10)
    Ar_Pars.Set_T(MD_Pars.kB, 300.0, 2)
    Ar_Pars.Non_Dim(Dim)
    print(Ar_Pars.mpVelocity)
    #
    Lib_Ar = []
    count = 0
    for i in range(Lib_N):
        new_Ar = Atom()
        count += 1
        new_Ar.ID = count
        new_Ar.Type = 1
        new_Ar.Elasticity = 0
        new_Ar.x = MD_Pars.BoxXLow + (MD_Pars.BoxXHigh - MD_Pars.BoxXLow) * random.random()
        new_Ar.y = MD_Pars.BoxYLow + (MD_Pars.BoxYHigh - MD_Pars.BoxYLow) * random.random()
        new_Ar.z = MD_Pars.BoxZHigh
        R1 = 0
        while R1 == 0:
            R1 = random.random()
        R2 = 0
        while R2 == 0:
            R2 = random.random()
        new_Ar.vx = Ar_Pars.mpVelocity * math.sqrt(- math.log(R1)) * math.cos(2 * MD_Pars.pi * R2)#Maxwell分布
        R1 = 0
        while R1 == 0:
            R1 = random.random()
        R2 = 0
        while R2 == 0:
            R2 = random.random()
        new_Ar.vy = Ar_Pars.mpVelocity * math.sqrt(- math.log(R1)) * math.sin(2 * MD_Pars.pi * R2)
        R1 = 0
        while R1 == 0:
            R1 = random.random()
        R2 = 0
        while R2 == 0:
            R2 = random.random()
        new_Ar.vz = - Ar_Pars.mpVelocity * math.sqrt(- math.log(R1))
        Lib_Ar.append(new_Ar)
    Lib_Dump(Lib_Ar)
    Dis_Vel('Ar', Lib_Ar, Lib_N, Ar_Pars.mpVelocity, 2, 0.0, 4.0, 0.0, 1.2)
    Dis_Pos('Ar', Lib_Ar, MD_Pars, Lib_N)
    #
    Pt_Wall = []
    count = 0
    for i in range(2 * Pt_I):
        for j in range(2 * Pt_J):
            for k in range(2 * Pt_K):
                if i / 2 + j / 2 + k / 2 == int(i / 2 + j / 2 + k / 2):
                    new_Pt = Atom()
                    count += 1
                    new_Pt.ID = count
                    new_Pt.Type = 2
                    new_Pt.Elasticity = 1
                    new_Pt.x = i / 2 * MD_Pars.fcc_lattice
                    new_Pt.y = j / 2 * MD_Pars.fcc_lattice
                    new_Pt.z = (k / 2 - 2.5) * MD_Pars.fcc_lattice
                    new_Pt.Bx = new_Pt.x
                    new_Pt.By = new_Pt.y
                    new_Pt.Bz = new_Pt.z
                    R1 = 0
                    while R1 == 0:
                        R1 = random.random()
                    R2 = 0
                    while R2 == 0:
                        R2 = random.random()
                    new_Pt.vx = Pt_Pars.mpVelocity / math.sqrt(3) * math.sqrt(- 2 * math.log(R1)) * math.cos(2 * MD_Pars.pi * R2)#高斯分布，平均值为0，方差=方均根速度用来控温
                    R1 = 0
                    while R1 == 0:
                        R1 = random.random()
                    R2 = 0
                    while R2 == 0:
                        R2 = random.random()
                    new_Pt.vy = Pt_Pars.mpVelocity / math.sqrt(3) * math.sqrt(- 2 * math.log(R1)) * math.cos(2 * MD_Pars.pi * R2)
                    R1 = 0
                    while R1 == 0:
                        R1 = random.random()
                    R2 = 0
                    while R2 == 0:
                        R2 = random.random()
                    new_Pt.vz = Pt_Pars.mpVelocity / math.sqrt(3) * math.sqrt(- 2 * math.log(R1)) * math.cos(2 * MD_Pars.pi * R2)
                    Pt_Wall.append(new_Pt)
    #
    Acceleration(Pt_Wall, Pt_Pars, MD_Pars, Pt_N)
    Rescale_T(Pt_Wall, Pt_Pars, MD_Pars, Dim, Pt_N)
    MD_Pars.MD_Dump(Pt_Wall)
    #
    Type = np.zeros(Pt_N)
    Elasticity = np.zeros(Pt_N)
    Pos = np.zeros((Pt_N, 6))
    Vel = np.zeros((Pt_N, 3))
    Acc = np.zeros((Pt_N, 6))
    Box = np.zeros((3, 2))
    ZLow = np.zeros(1)
    LJ_E = np.zeros(3)
    LJ_S = np.zeros(3)
    Mass = np.zeros(2)
    Pt_argVel = np.zeros(3)
    Pt_V2 = np.zeros(1)
    Pt_T = np.zeros(1)
    T = np.zeros(2)
    for i in range(Pt_N):
        Type[i] = Pt_Wall[i].Type
        Elasticity[i] = Pt_Wall[i].Elasticity
        Pos[i, 0] = Pt_Wall[i].x
        Pos[i, 1] = Pt_Wall[i].y
        Pos[i, 2] = Pt_Wall[i].z
        Pos[i, 3] = Pt_Wall[i].Bx
        Pos[i, 4] = Pt_Wall[i].By
        Pos[i, 5] = Pt_Wall[i].Bz
        Vel[i, 0] = Pt_Wall[i].vx
        Vel[i, 1] = Pt_Wall[i].vy
        Vel[i, 2] = Pt_Wall[i].vz
        Acc[i, 0] = Pt_Wall[i].ax
        Acc[i, 1] = Pt_Wall[i].ay
        Acc[i, 2] = Pt_Wall[i].az
        Box[0, 0] = MD_Pars.BoxXLow
        Box[0, 1] = MD_Pars.BoxXHigh
        Box[1, 0] = MD_Pars.BoxYLow
        Box[1, 1] = MD_Pars.BoxYHigh
        Box[2, 0] = MD_Pars.BoxZLow
        Box[2, 1] = MD_Pars.BoxZHigh
        ZLow[0] = np.nan
        LJ_E[0] = Ar_Pars.Epsilon
        LJ_E[1] = Pt_Pars.Epsilon
        LJ_E[2] = math.sqrt(LJ_E[0] * LJ_E[1])
        LJ_S[0] = Ar_Pars.Sigma
        LJ_S[1] = Pt_Pars.Sigma
        LJ_S[2] = (LJ_S[0] + LJ_S[1]) / 2
        Mass[0] = Ar_Pars.Mass
        Mass[1] = Pt_Pars.Mass
        Pt_argVel[0] = 0.0
        Pt_argVel[1] = 0.0
        Pt_argVel[2] = 0.0
        Pt_V2[0] = 0.0
        T[0] = Ar_Pars.T
        T[1] = Pt_Pars.T
    d_Type = cuda.to_device(Type)
    d_Elasticity = cuda.to_device(Elasticity)
    d_Pos = cuda.to_device(Pos)
    d_Vel = cuda.to_device(Vel)
    d_Acc = cuda.to_device(Acc)
    BD = int((Pt_N + 511) / 512)
    #
    start = time.time()
    while MD_Pars.State:
        TA.Time_Advancement[BD, 512](d_Type, d_Elasticity, d_Pos, d_Vel, d_Acc, Box, ZLow, LJ_E, LJ_S, Mass, Pt_argVel, Pt_V2, Pt_T, T, Pt_N, MD_Pars.dt, MD_Pars.cutoff, MD_Pars.Spr_K, Dim.Velocity, Dim.Mass, MD_Pars.kB, Pt_N)
        #TA.Time_Advancement1[BD,512](d_Type,d_Elasticity,d_Pos,d_Vel,d_Acc,Box,ZLow,LJ_E,LJ_S,Mass,Pt_argVel,Pt_V2,Pt_T,T,Pt_N,MD_Pars.dt,MD_Pars.cutoff,MD_Pars.Spr_K,Dim.Velocity,Dim.Mass,MD_Pars.kB,Pt_N)
        #Box[2,0]=ZLow[0]
        #TA.Time_Advancement2[BD,512](d_Type,d_Elasticity,d_Pos,d_Vel,d_Acc,Box,ZLow,LJ_E,LJ_S,Mass,Pt_argVel,Pt_V2,Pt_T,T,Pt_N,MD_Pars.dt,MD_Pars.cutoff,MD_Pars.Spr_K,Dim.Velocity,Dim.Mass,MD_Pars.kB,Pt_N)
        #TA.Time_Advancement3[BD,512](d_Type,d_Elasticity,d_Pos,d_Vel,d_Acc,Box,ZLow,LJ_E,LJ_S,Mass,Pt_argVel,Pt_V2,Pt_T,T,Pt_N,MD_Pars.dt,MD_Pars.cutoff,MD_Pars.Spr_K,Dim.Velocity,Dim.Mass,MD_Pars.kB,Pt_N)
        #TA.Time_Advancement4[BD,512](d_Type,d_Elasticity,d_Pos,d_Vel,d_Acc,Box,ZLow,LJ_E,LJ_S,Mass,Pt_argVel,Pt_V2,Pt_T,T,Pt_N,MD_Pars.dt,MD_Pars.cutoff,MD_Pars.Spr_K,Dim.Velocity,Dim.Mass,MD_Pars.kB,Pt_N)
        #
        Pos = d_Pos.copy_to_host()
        Vel = d_Vel.copy_to_host()
        Acc = d_Acc.copy_to_host()
        for i in range(Pt_N):
            Pt_Wall[i].x = Pos[i, 0]
            Pt_Wall[i].y = Pos[i, 1]
            Pt_Wall[i].z = Pos[i, 2]
            Pt_Wall[i].Bx = Pos[i, 3]
            Pt_Wall[i].By = Pos[i, 4]
            Pt_Wall[i].Bz = Pos[i, 5]
            Pt_Wall[i].vx = Vel[i, 0]
            Pt_Wall[i].vy = Vel[i, 1]
            Pt_Wall[i].vz = Vel[i, 2]
            Pt_Wall[i].ax = Acc[i, 0]
            Pt_Wall[i].ay = Acc[i, 1]
            Pt_Wall[i].az = Acc[i, 2]
        #
        ZLow[0] = np.nan
        Pt_argVel[0] = 0.0
        Pt_argVel[1] = 0.0
        Pt_argVel[2] = 0.0
        Pt_V2[0] = 0.0
        #
        MD_Pars.Period_WallT.append(Pt_T[0])
        MD_Pars.TimeStep += 1
        MD_Pars.MD_Dump(Pt_Wall)
    end = time.time()
    print('Relaxation Done in ' + str(round(end - start, 10)) + ' seconds!')
    #
    #gif_name='Pt_D_v.gif'
    #path=r'C:\Users\user\Desktop\Init'
    #fps=5
    #PNGname=['Pt_'+str(i)+'_D_v.png' for i in range(0,MD_Pars.Period+1,Gif_Step)]
    #CG.Create_Gif(gif_name, path, fps, PNGname)
    #
    fig = mpl.figure()
    lab = 'T of Pt Wall in Relaxation'
    TS = [ts * MD_Pars.dt for ts in range(MD_Pars.Period + 1)]
    mpl.plot(TS, MD_Pars.Period_WallT, 'ro', markersize = 1, label = lab)
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$t$')
    mpl.ylabel('$T (K)$')
    mpl.axis([0.0, MD_Pars.Period * MD_Pars.dt, 200.0, 400.0])
    mpl.savefig('Pt_T_Relaxation.png', dpi = 600)
    #
    #mpl.show()
    mpl.close()
