'''
Name:              Molecular Beam.py
Description:       To verify molecular beam experiments
Auther:            KhalilWong
License:           MIT License
Copyright:         Copyright (C) 2020, by KhalilWong
Version:           2.0.0
Date:              2020/03/08
Namespace:         https://github.com/KhalilWong
DownloadURL:       https://github.com/KhalilWong/...
UpdateURL:         https://github.com/KhalilWong/...
'''
################################################################################
import matplotlib.pyplot as mpl
import numpy as np
import numba as nb
import Distribution_Cloud_test_new as DC

################################################################################
@nb.jit(nopython = True, nogil = True)
def CLL_MB(Model, u, w):
    #
    N = 1000
    ur = np.zeros(N + 1)
    vr = np.zeros(N + 1)
    wr = np.zeros(N + 1)
    fur = np.zeros(N + 1)
    fvr = np.zeros(N + 1)
    fwr = np.zeros(N + 1)
    ur_Low = - 3.0
    ur_High = 3.0
    vr_Low = - 3.0
    vr_High = 3.0
    wr_Low = 0.0
    wr_High = 3.0
    #
    Type = 0
    l1 = 1.194
    l2 = 1.114
    l3 = 0.431
    #
    for i in range(N + 1):
        ur[i] = (ur_High - ur_Low) * i / N + ur_Low
        vr[i] = (vr_High - vr_Low) * i / N + vr_Low
        fur[i] = Model(Type, u / np.sqrt(2), u / np.sqrt(2), w, ur[i], l1, l2, l3)
        fvr[i] = Model(Type, u / np.sqrt(2), u / np.sqrt(2), w, vr[i], l1, l2, l3)
    #
    Type = 1
    l1 = 0.964
    l2 = 0.970
    l3 = 0.0
    #
    for i in range(N + 1):
        wr[i] = (wr_High - wr_Low) * i / N + wr_Low
        fwr[i] = Model(Type, w, u / np.sqrt(2), u / np.sqrt(2), wr[i], l1, l2, l3)
    #
    return(ur, fur, vr, fvr, wr, fwr)

################################################################################
@nb.jit(nopython = True, nogil = True)
def CLL(Type, ui, vi, wi, ur, l1, l2, l3):
    if Type == 0:
        sigma = l1 * np.exp(- l2 * vi ** 2 - l3 * wi ** 2)# - 0.175)
        R = 1 / np.sqrt(np.pi * sigma * (2 - sigma)) * np.exp(- (ur - (1 - sigma) * ui) ** 2 / (sigma * (2 - sigma)))
    elif Type == 1:
        alpha = l1 * np.exp(- l2 * vi ** 2 - l2 * wi ** 2)
        N_theta = 1000
        R = 0.0
        for i in range(N_theta):
            R += 2 * ur / alpha * np.exp(- (ur ** 2 + (1 - alpha) * ui ** 2 - 2 * np.sqrt(1 - alpha) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / alpha) / N_theta
    #
    return(R)

################################################################################
@nb.jit(nopython = True, nogil = True)
def EnTAC(u, ur, fur):
    #
    arg_ur = 0.0
    #arg_ur2 = 0.0
    for i in range(len(ur)):
        arg_ur += ur[i] * fur[i] * (ur[1] - ur[0])
        #arg_ur2 += (ur[i] ** 2) * fur[i] * (ur[1] - ur[0])
    #
    uw = 1.0
    EAC = np.abs((u ** 2 - arg_ur ** 2) / (u ** 2 - uw ** 2))
    TMAC = np.abs((u - arg_ur) / (u - uw))
    #
    return(EAC, TMAC)

################################################################################
@nb.jit(nopython = True, nogil = True)
def EnTAC_EnTheta(m, ND_Velocity):
    #
    N = 25
    ELow = 162
    EHigh = 1458
    E = np.zeros(N + 1)
    TheLow = 0
    TheHigh = 67.5
    Theta = np.zeros(N + 1)
    for i in range(N + 1):
        E[i] = ELow + (EHigh - ELow) * i / N
        Theta[i] = TheLow + (TheHigh - TheLow) * i / N
    #
    EACx_E = np.zeros(N + 1)
    TMACx_E = np.zeros(N + 1)
    EACy_E = np.zeros(N + 1)
    TMACy_E = np.zeros(N + 1)
    EACz_E = np.zeros(N + 1)
    TMACz_E = np.zeros(N + 1)
    for i in range(N + 1):
        #
        theta = 22.5
        u, w = Get_MBVelocity(E[i], m, theta)
        u /= (ND_Velocity * 0.88)
        w /= (ND_Velocity * 0.88)
        ur, fur, vr, fvr, wr, fwr = CLL_MB(CLL, u, w)
        EACx_E[i], TMACx_E[i] = EnTAC(u / np.sqrt(2), ur, fur)
        EACy_E[i], TMACy_E[i] = EnTAC(u / np.sqrt(2), vr, fvr)
        EACz_E[i], TMACz_E[i] = EnTAC(w, wr, fwr)
    #
    EACx_The = np.zeros(N + 1)
    TMACx_The = np.zeros(N + 1)
    EACy_The = np.zeros(N + 1)
    TMACy_The = np.zeros(N + 1)
    EACz_The = np.zeros(N + 1)
    TMACz_The = np.zeros(N + 1)
    for i in range(N + 1):
        #
        t = 450
        u, w = Get_MBVelocity(t, m, Theta[i])
        u /= (ND_Velocity * 0.88)
        w /= (ND_Velocity * 0.88)
        ur, fur, vr, fvr, wr, fwr = CLL_MB(CLL, u, w)
        EACx_The[i], TMACx_The[i] = EnTAC(u / np.sqrt(2), ur, fur)
        EACy_The[i], TMACy_The[i] = EnTAC(u / np.sqrt(2), vr, fvr)
        EACz_The[i], TMACz_The[i] = EnTAC(w, wr, fwr)
    #
    return(E, EACx_E, EACy_E, EACz_E, TMACx_E, Theta, EACx_The, EACy_The, EACz_The, TMACx_The)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Maxwell(u):
    #
    #f = np.sqrt(1 / np.pi) * np.exp(- u ** 2)
    f = 2 * abs(u) * np.exp(- u ** 2)
    #
    return(f)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Get_MBVelocity(T, m, theta):
    #
    kB = 1.380649e-23
    E = T * kB
    theta = theta * np.pi / 180
    u = np.sqrt(2 * E / m) * np.sin(theta)
    w = - np.sqrt(2 * E / m) * np.cos(theta)
    #
    return(u, w)

################################################################################
@nb.jit(nopython = True, nogil = True)
def MD_MB_CountEnThe(VX, VY, VZ, m, ND_Velocity, kB):
    #
    N = len(VX)
    EdkB = np.zeros(N)
    THETA = np.zeros(N)
    for i in range(N):
        EdkB[i] = (VX[i] ** 2 + VY[i] ** 2 + VZ[i] ** 2) * ND_Velocity ** 2 * m * 0.5 / kB
        THETA[i] = np.arctan(np.sqrt((VX[i] ** 2 + VY[i] ** 2) / (VZ[i] ** 2)))
    #
    THETA *= 180 / np.pi
    return(EdkB, THETA)

################################################################################
@nb.jit(nopython = True, nogil = True)
def MD_MB_FIG14(VX, VY, FVX, FVY, EdkB, THETA):
    #
    E = 450
    N = len(VX)
    theta = np.array([0, 22.5, 45.0, 67.5, 90.0])
    M = len(theta)
    #
    ur_Low = - 3.0
    ur_High = 3.0
    K = 20
    dur = (ur_High - ur_Low) / K
    dE = 250.0
    dtheta = 6.0
    ur = np.zeros(K + 1)
    ur_Count = np.zeros(M)
    ur_mat = np.zeros((M, K + 1))
    for i in range(K + 1):
        ur[i] = ur_Low + (ur_High - ur_Low) * i / K
    #
    for i in range(N):
        if E - dE <= EdkB[i] <= E + dE:
            for j in range(M):
                if theta[j] - dtheta <= THETA[i] <= theta[j] + dtheta:
                    #
                    UR = VX[i] * FVX[i] / np.abs(VX[i])
                    #UR = (VX[i] * FVX[i] + VY[i] * FVY[i]) / np.sqrt(VX[i] ** 2 + VY[i] ** 2)
                    k = int((UR - ur_Low) / dur + 0.5)
                    if 0 <= k <= K:
                        ur_Count[j] += 1
                        ur_mat[j, k] += 1
                        break
    #
    Count = 0.0
    for j in range(M):
        Count += ur_Count[j]
        if ur_Count[j] != 0:
            ur_mat[j, :] /= ur_Count[j] * dur
    #
    return(ur, ur_mat, Count)

################################################################################
@nb.jit(nopython = True, nogil = True)
def MD_MB_FIG11(VX, VY, VZ, FVX, FVY, FVZ, EdkB, THETA):
    #
    N = 25
    ELow = 162
    EHigh = 1458
    E = np.zeros(N + 1)
    EAC_Ex = np.zeros(N + 1)
    EAC_Ey = np.zeros(N + 1)
    EAC_Ez = np.zeros(N + 1)
    TMAC_E = np.zeros(N + 1)
    TheLow = 0
    TheHigh = 67.5
    Theta = np.zeros(N + 1)
    EAC_Thex = np.zeros(N + 1)
    EAC_They = np.zeros(N + 1)
    EAC_Thez = np.zeros(N + 1)
    TMAC_The = np.zeros(N + 1)
    for i in range(N + 1):
        E[i] = ELow + (EHigh - ELow) * i / N
        Theta[i] = TheLow + (TheHigh - TheLow) * i / N
    #
    N = len(VX)
    uw = 0.88 * 1.0
    dE = 100.0
    dtheta = 3.0
    #
    theta = 22.5
    M = len(E)
    Sum_Eix = np.zeros(M)
    Sum_Eiy = np.zeros(M)
    Sum_Eiz = np.zeros(M)
    Sum_Erx = np.zeros(M)
    Sum_Ery = np.zeros(M)
    Sum_Erz = np.zeros(M)
    Sum_Ew = np.zeros(M)
    Sum_ui = np.zeros(M)
    Sum_ur = np.zeros(M)
    Sum_vi = np.zeros(M)
    Sum_vr = np.zeros(M)
    Sum_wi = np.zeros(M)
    Sum_wr = np.zeros(M)
    Sum_uw = np.zeros(M)
    for i in range(N):
        if theta - dtheta <= THETA[i] <= theta + dtheta:
            for j in range(M):
                if E[j] - dE <= EdkB[i] <= E[j] + dE:
                    #
                    Sum_Eix[j] += VX[i] ** 2
                    #Sum_Eix[j] += VX[i] ** 2 + VY[i] ** 2
                    Sum_Eiy[j] += VY[i] ** 2
                    #Sum_Eiy[j] += 0.0
                    Sum_Eiz[j] += VZ[i] ** 2
                    Sum_Erx[j] += FVX[i] ** 2
                    #Sum_Erx[j] += (VX[i] * FVX[i] + VY[i] * FVY[i]) ** 2 / (VX[i] ** 2 + VY[i] ** 2)
                    Sum_Ery[j] += FVY[i] ** 2
                    #Sum_Ery[j] += (VX[i] * FVY[i] - VY[i] * FVX[i]) ** 2 / (VX[i] ** 2 + VY[i] ** 2)
                    Sum_Erz[j] += FVZ[i] ** 2
                    Sum_Ew[j] += uw ** 2
                    #
                    #Sum_ui[j] += np.sqrt(VX[i] ** 2 + VY[i] ** 2)
                    Sum_ui[j] += np.abs(VX[i])
                    #Sum_ui[j] += np.abs(VX[i])
                    #Sum_ur[j] += (VX[i] * FVX[i] + VY[i] * FVY[i]) / np.sqrt(VX[i] ** 2 + VY[i] ** 2)
                    Sum_ur[j] += VX[i] * FVX[i] / np.abs(VX[i])
                    #Sum_ur[j] += np.abs(FVX[i])
                    Sum_vi[j] += np.abs(VY[i])
                    #Sum_vi[j] += 0.0
                    Sum_vr[j] += VY[i] * FVY[i] / np.abs(VY[i])
                    #Sum_vr[j] += (VX[i] * FVY[i] - VY[i] * FVX[i]) / np.sqrt(VX[i] ** 2 + VY[i] ** 2)
                    Sum_wi[j] += VZ[i]
                    Sum_wr[j] += FVZ[i]
                    Sum_uw[j] += uw
                    #EAC_Ex[j] = np.abs((Sum_Eix[j] - Sum_Erx[j]) / (Sum_Eix[j] - Sum_Ew[j]))
                    EAC_Ex[j] = np.abs((Sum_ui[j] ** 2 - Sum_ur[j] ** 2) / (Sum_ui[j] ** 2 - Sum_uw[j] ** 2))
                    #EAC_Ey[j] = np.abs((Sum_Eiy[j] - Sum_Ery[j]) / (Sum_Eiy[j] - Sum_Ew[j]))
                    EAC_Ey[j] = np.abs((Sum_vi[j] ** 2 - Sum_vr[j] ** 2) / (Sum_vi[j] ** 2 - Sum_uw[j] ** 2))
                    #EAC_Ez[j] = np.abs((Sum_Eiz[j] - Sum_Erz[j]) / (Sum_Eiz[j] - Sum_Ew[j]))
                    EAC_Ez[j] = np.abs((Sum_wi[j] ** 2 - Sum_wr[j] ** 2) / (Sum_wi[j] ** 2 - Sum_uw[j] ** 2))
                    TMAC_E[j] = np.abs((Sum_ui[j] - Sum_ur[j]) / (Sum_ui[j] - Sum_uw[j]))
                    break
    #
    e = 450
    M = len(Theta)
    Sum_Eix = np.zeros(M)
    Sum_Eiy = np.zeros(M)
    Sum_Eiz = np.zeros(M)
    Sum_Erx = np.zeros(M)
    Sum_Ery = np.zeros(M)
    Sum_Erz = np.zeros(M)
    Sum_Ew = np.zeros(M)
    Sum_ui = np.zeros(M)
    Sum_ur = np.zeros(M)
    Sum_vi = np.zeros(M)
    Sum_vr = np.zeros(M)
    Sum_wi = np.zeros(M)
    Sum_wr = np.zeros(M)
    Sum_uw = np.zeros(M)
    for i in range(N):
        if e - dE <= EdkB[i] <= e + dE:
            for j in range(M):
                if Theta[j] - dtheta <= THETA[i] <= Theta[j] + dtheta:
                    #
                    Sum_Eix[j] += VX[i] ** 2
                    #Sum_Eix[j] += VX[i] ** 2 + VY[i] ** 2
                    Sum_Eiy[j] += VY[i] ** 2
                    #Sum_Eiy[j] += 0.0
                    Sum_Eiz[j] += VZ[i] ** 2
                    Sum_Erx[j] += FVX[i] ** 2
                    #Sum_Erx[j] += (VX[i] * FVX[i] + VY[i] * FVY[i]) ** 2 / (VX[i] ** 2 + VY[i] ** 2)
                    Sum_Ery[j] += FVY[i] ** 2
                    #Sum_Ery[j] += (VX[i] * FVY[i] - VY[i] * FVX[i]) ** 2 / (VX[i] ** 2 + VY[i] ** 2)
                    Sum_Erz[j] += FVZ[i] ** 2
                    Sum_Ew[j] += uw ** 2
                    #
                    #Sum_ui[j] += np.sqrt(VX[i] ** 2 + VY[i] ** 2)
                    Sum_ui[j] += np.abs(VX[i])
                    #Sum_ui[j] += np.abs(VX[i])
                    #Sum_ur[j] += (VX[i] * FVX[i] + VY[i] * FVY[i]) / np.sqrt(VX[i] ** 2 + VY[i] ** 2)
                    Sum_ur[j] += VX[i] * FVX[i] / np.abs(VX[i])
                    #Sum_ur[j] += np.abs(FVX[i])
                    Sum_vi[j] += np.abs(VY[i])
                    #Sum_vi[j] += 0.0
                    Sum_vr[j] += VY[i] * FVY[i] / np.abs(VY[i])
                    #Sum_vr[j] += (VX[i] * FVY[i] - VY[i] * FVX[i]) / np.sqrt(VX[i] ** 2 + VY[i] ** 2)
                    Sum_wi[j] += VZ[i]
                    Sum_wr[j] += FVZ[i]
                    Sum_uw[j] += uw
                    #EAC_Thex[j] = np.abs((Sum_Eix[j] - Sum_Erx[j]) / (Sum_Eix[j] - Sum_Ew[j]))
                    EAC_Thex[j] = np.abs((Sum_ui[j] ** 2 - Sum_ur[j] ** 2) / (Sum_ui[j] ** 2 - Sum_uw[j] ** 2))
                    #EAC_They[j] = np.abs((Sum_Eiy[j] - Sum_Ery[j]) / (Sum_Eiy[j] - Sum_Ew[j]))
                    EAC_They[j] = np.abs((Sum_vi[j] ** 2 - Sum_vr[j] ** 2) / (Sum_vi[j] ** 2 - Sum_uw[j] ** 2))
                    #EAC_Thez[j] = np.abs((Sum_Eiz[j] - Sum_Erz[j]) / (Sum_Eiz[j] - Sum_Ew[j]))
                    EAC_Thez[j] = np.abs((Sum_wi[j] ** 2 - Sum_wr[j] ** 2) / (Sum_wi[j] ** 2 - Sum_uw[j] ** 2))
                    TMAC_The[j] = np.abs((Sum_ui[j] - Sum_ur[j]) / (Sum_ui[j] - Sum_uw[j]))
                    break
    #
    return(E, EAC_Ex, EAC_Ey, EAC_Ez, TMAC_E, Theta, EAC_Thex, EAC_They, EAC_Thez, TMAC_The)

################################################################################
def main():
    #
    T = 450
    kB = 1.380649e-23
    amu = 1.66053886e-27
    m = 39.95 * amu
    ND_Velocity = 400.886
    #
    dtheta = 1
    N = int(180 / dtheta)
    maxur = np.zeros((N + 1, 3))
    for i in range(N + 1):
        maxur[i, 0] = i * dtheta
        u, w = Get_MBVelocity(T, m, i * dtheta)
        u /= (ND_Velocity * 0.88)
        w /= (ND_Velocity * 0.88)
        ur, fur, vr, fvr, wr, fwr = CLL_MB(CLL, u, w)
        MaxIndex = np.argmax(fur)
        maxur[i, 1] = ur[MaxIndex]
        maxur[i, 2] = u / np.sqrt(2)
    fig = mpl.figure()
    mpl.plot(maxur[:, 0], maxur[:, 1], 'r-', markersize = 5)
    mpl.plot(maxur[:, 0], maxur[:, 2], 'b-', markersize = 5)
    #mpl.show()
    mpl.savefig('MB_FIG14_bonus.png', dpi = 600)
    mpl.close()
    #mpl.close()
    #
    u1, w1 = Get_MBVelocity(T, m, 0.0)
    u1 /= (ND_Velocity * 0.88)
    w1 /= (ND_Velocity * 0.88)
    u2, w2 = Get_MBVelocity(T, m, 22.5)
    u2 /= (ND_Velocity * 0.88)
    w2 /= (ND_Velocity * 0.88)
    u3, w3 = Get_MBVelocity(T, m, 45.0)
    u3 /= (ND_Velocity * 0.88)
    w3 /= (ND_Velocity * 0.88)
    u4, w4 = Get_MBVelocity(T, m, 67.5)
    u4 /= (ND_Velocity * 0.88)
    w4 /= (ND_Velocity * 0.88)
    u5, w5 = Get_MBVelocity(T, m, 90.0)
    u5 /= (ND_Velocity * 0.88)
    w5 /= (ND_Velocity * 0.88)
    '''
    u6, w6 = Get_MBVelocity(T, m, 75.0)
    u6 /= (ND_Velocity * 0.88)
    w6 /= (ND_Velocity * 0.88)
    u7, w7 = Get_MBVelocity(T, m, 90.0)
    u7 /= (ND_Velocity * 0.88)
    w7 /= (ND_Velocity * 0.88)
    '''
    #
    ur1, fur1, vr1, fvr1, wr1, fwr1 = CLL_MB(CLL, u1, w1)
    ur2, fur2, vr2, fvr2, wr2, fwr2 = CLL_MB(CLL, u2, w2)
    ur3, fur3, vr3, fvr3, wr3, fwr3 = CLL_MB(CLL, u3, w3)
    ur4, fur4, vr4, fvr4, wr4, fwr4 = CLL_MB(CLL, u4, w4)
    ur5, fur5, vr5, fvr5, wr5, fwr5 = CLL_MB(CLL, u5, w5)
    #ur6, fur6, vr6, fvr6, wr6, fwr6 = CLL_MB(CLL, u6, w6)
    #ur7, fur7, vr7, fvr7, wr7, fwr7 = CLL_MB(CLL, u7, w7)
    #
    '''
    fig = mpl.figure()
    mpl.plot(ur1, fur1, 'b--', markersize = 5, label = '0')
    mpl.plot(ur2, fur2, 'g--', markersize = 5, label = '15')
    mpl.plot(ur3, fur3, 'r--', markersize = 5, label = '30')
    mpl.plot(ur4, fur4, 'c--', markersize = 5, label = '45')
    mpl.plot(ur5, fur5, 'm--', markersize = 5, label = '60')
    mpl.plot(ur6, fur6, 'y--', markersize = 5, label = '75')
    mpl.plot(ur7, fur7, 'k--', markersize = 5, label = '90')
    #mpl.plot(u[:, 1], fu, 'w-', markersize = 5, label = 'fui')
    #mpl.plot(w[:, 1], fw, 'g--', markersize = 5, label = 'fwi')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$u_x$')
    mpl.ylabel('$f$')
    #mpl.axis([-3.0, 3.0, 0.0, 0.6])
    mpl.savefig('MB_FIG14.png', dpi = 600)
    #mpl.show()
    mpl.close()
    '''
    #
    E, EACx_E, EACy_E, EACz_E, TMACx_E, Theta, EACx_The, EACy_The, EACz_The, TMACx_The = EnTAC_EnTheta(m, ND_Velocity)
    '''
    fig1 = mpl.figure()
    mpl.plot(E, EACx_E, 'r--', markersize = 5, label = 'EAC-x')
    mpl.plot(E, EACy_E, 'b--', markersize = 5, label = 'EAC-y')
    mpl.plot(E, EACz_E, 'g--', markersize = 5, label = 'EAC-z')
    mpl.plot(E, TMACx_E, 'y--', markersize = 5, label = 'TMAC')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$E/k$')
    mpl.ylabel('$AC$')
    mpl.axis([0.0, 1500.0, 0.0, 5.0])
    mpl.savefig('MB_FIG10.png', dpi = 600)
    mpl.close()
    fig1 = mpl.figure()
    mpl.plot(Theta, EACx_The, 'r--', markersize = 5, label = 'EAC-x')
    mpl.plot(Theta, EACy_The, 'b--', markersize = 5, label = 'EAC-y')
    mpl.plot(Theta, EACz_The, 'g--', markersize = 5, label = 'EAC-z')
    mpl.plot(Theta, TMACx_The, 'y--', markersize = 5, label = 'TMAC')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$Theta$')
    mpl.ylabel('$AC$')
    mpl.axis([0.0, 70.0, 0.0, 15.0])
    mpl.savefig('MB_FIG11.png', dpi = 600)
    mpl.close()
    fig1 = mpl.figure()
    mpl.plot(EACx_E, TMACx_E, 'bo', markersize = 5, label = '$\Theta = 22.5$')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$EAC-x$')
    mpl.ylabel('$TMAC$')
    mpl.axis([0.0, 1.0, 0.0, 1.0])
    mpl.savefig('MB_FIG12.png', dpi = 600)
    mpl.close()
    '''
    #
    FileName = 'Incident_Reflection.data'
    ID, Tt, X, Y, Z, VX, VY, VZ, FX, FY, FZ, FVX, FVY, FVZ, Adsorbed = DC.ReadFile(FileName)
    GasT = 300.0
    WallT = 300.0
    GasTGasmpV = np.sqrt(2 * 1.38E-23 * GasT / (39.95 / 6.02 * 1E-26))
    WallTGasmpV = np.sqrt(2 * 1.38E-23 * WallT / (39.95 / 6.02 * 1E-26))        #!=WallTWallmpV
    GasTGasmpV /= np.sqrt(5.207E-20 / (195.08 / 6.02 * 1E-26))
    WallTGasmpV /= np.sqrt(5.207E-20 / (195.08 / 6.02 * 1E-26))
    print(GasTGasmpV, WallTGasmpV)
    VX /= GasTGasmpV
    VY /= GasTGasmpV
    VZ /= GasTGasmpV
    FVX /= GasTGasmpV
    FVY /= GasTGasmpV
    FVZ /= GasTGasmpV
    EdkB, THETA = MD_MB_CountEnThe(VX * GasTGasmpV, VY * GasTGasmpV, VZ * GasTGasmpV, m, ND_Velocity, kB)
    print(np.mean(EdkB))
    N = len(EdkB)
    with open('EnTHETA.dat', 'w') as out:
        for i in range(N):
            print('%f\t%f' % (EdkB[i], THETA[i]), file = out)
    #ur, ur_mat, Count = MD_MB_FIG14(VX * GasTGasmpV, VY * GasTGasmpV, FVX * GasTGasmpV, FVY * GasTGasmpV, EdkB, THETA)
    ur, ur_mat, Count = MD_MB_FIG14(VX, VY, FVX, FVY, EdkB, THETA)
    print(Count)
    fig14 = mpl.figure()
    mpl.plot(ur, ur_mat[0, :], 'b-', markersize = 3, label = '0')
    mpl.plot(ur, ur_mat[1, :], 'g-', markersize = 3, label = '22.5')
    mpl.plot(ur, ur_mat[2, :], 'r-', markersize = 3, label = '45')
    mpl.plot(ur, ur_mat[3, :], 'c-', markersize = 3, label = '67.5')
    mpl.plot(ur, ur_mat[4, :], 'm-', markersize = 3, label = '90')
    mpl.plot(ur1, fur1, 'b--', markersize = 5, label = '0')
    mpl.plot(ur2, fur2, 'g--', markersize = 5, label = '22.5')
    mpl.plot(ur3, fur3, 'r--', markersize = 5, label = '45')
    mpl.plot(ur4, fur4, 'c--', markersize = 5, label = '67.5')
    mpl.plot(ur5, fur5, 'm--', markersize = 5, label = '90')
    with open('FIG14_MCLL.dat', 'w') as out:
        print('%s\t%s\t%s\t%s\t%s\t%s' % ('ur', 'fur1', 'fur2', 'fur3', 'fur4', 'fur5'), file = out)
        for i in range(len(ur1)):
            print('%f\t%f\t%f\t%f\t%f\t%f' % (ur1[i], fur1[i], fur2[i], fur3[i], fur4[i], fur5[i]), file = out)
    with open('FIG14_MD.dat', 'w') as out:
        print('%s\t%s\t%s\t%s\t%s\t%s' % ('ur', 'fur1', 'fur2', 'fur3', 'fur4', 'fur5'), file = out)
        for i in range(len(ur)):
            print('%f\t%f\t%f\t%f\t%f\t%f' % (ur[i], ur_mat[0, i], ur_mat[1, i], ur_mat[2, i], ur_mat[3, i], ur_mat[4, i]), file = out)
    #mpl.plot(ur6, fur6, 'y--', markersize = 5, label = '75')
    #mpl.plot(ur7, fur7, 'k--', markersize = 5, label = '90')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$ur$')
    mpl.ylabel('$f$')
    mpl.savefig('MB_FIG14.png', dpi = 600)
    mpl.close()
    mpl.show()
    #
    E, EAC_Ex, EAC_Ey, EAC_Ez, TMAC_E, Theta, EAC_Thex, EAC_They, EAC_Thez, TMAC_The = MD_MB_FIG11(VX * GasTGasmpV, VY * GasTGasmpV, VZ * GasTGasmpV, FVX * GasTGasmpV, FVY * GasTGasmpV, FVZ * GasTGasmpV, EdkB, THETA)
    fig1 = mpl.figure()
    mpl.plot(E, EAC_Ex, 'ro', markersize = 5, label = 'EAC-x-MD')
    mpl.plot(E, EAC_Ey, 'bo', markersize = 5, label = 'EAC-y-MD')
    mpl.plot(E, EAC_Ez, 'go', markersize = 5, label = 'EAC-z-MD')
    mpl.plot(E, TMAC_E, 'yo', markersize = 5, label = 'TMAC-MD')
    mpl.plot(E, EACx_E, 'r--', markersize = 5, label = 'EAC-x')
    mpl.plot(E, EACy_E, 'b--', markersize = 5, label = 'EAC-y')
    mpl.plot(E, EACz_E, 'g--', markersize = 5, label = 'EAC-z')
    mpl.plot(E, TMACx_E, 'y--', markersize = 5, label = 'TMAC')
    with open('FIG10_MD-MCLL.dat', 'w') as out:
        print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('E', 'EACX-MD', 'EACY-MD', 'EACZ-MD', 'TMAC-MD', 'EACX-MCLL', 'EACY-MCLL', 'EACZ-MCLL', 'TMAC-MCLL'), file = out)
        for i in range(len(E)):
            print('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f' % (E[i], EAC_Ex[i], EAC_Ey[i], EAC_Ez[i], TMAC_E[i], EACx_E[i], EACy_E[i], EACz_E[i], TMACx_E[i]), file = out)
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$E/k$')
    mpl.ylabel('$AC$')
    mpl.axis([0.0, 1500.0, 0.0, 2.0])
    mpl.savefig('MB_FIG10.png', dpi = 600)
    mpl.close()
    fig1 = mpl.figure()
    mpl.plot(Theta, EAC_Thex, 'ro', markersize = 5, label = 'EAC-x-MD')
    mpl.plot(Theta, EAC_They, 'bo', markersize = 5, label = 'EAC-y-MD')
    mpl.plot(Theta, EAC_Thez, 'go', markersize = 5, label = 'EAC-z-MD')
    mpl.plot(Theta, TMAC_The, 'yo', markersize = 5, label = 'TMAC-MD')
    mpl.plot(Theta, EACx_The, 'r--', markersize = 5, label = 'EAC-x')
    mpl.plot(Theta, EACy_The, 'b--', markersize = 5, label = 'EAC-y')
    mpl.plot(Theta, EACz_The, 'g--', markersize = 5, label = 'EAC-z')
    mpl.plot(Theta, TMACx_The, 'y--', markersize = 5, label = 'TMAC')
    with open('FIG11_MD-MCLL.dat', 'w') as out:
        print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('Theta', 'EACX-MD', 'EACY-MD', 'EACZ-MD', 'TMAC-MD', 'EACX-MCLL', 'EACY-MCLL', 'EACZ-MCLL', 'TMAC-MCLL'), file = out)
        for i in range(len(Theta)):
            print('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f' % (Theta[i], EAC_Thex[i], EAC_They[i], EAC_Thez[i], TMAC_The[i], EACx_The[i], EACy_The[i], EACz_The[i], TMACx_The[i]), file = out)
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$Theta$')
    mpl.ylabel('$AC$')
    mpl.axis([0.0, 70.0, 0.0, 4.0])
    mpl.savefig('MB_FIG11.png', dpi = 600)
    mpl.close()
    fig1 = mpl.figure()
    mpl.plot(EAC_Ex, TMAC_E, 'ro', markersize = 5, label = '$\Theta = 22.5-MD$')
    mpl.plot(EACx_E, TMACx_E, 'bo', markersize = 5, label = '$\Theta = 22.5$')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$EAC-x$')
    mpl.ylabel('$TMAC$')
    mpl.axis('square')
    mpl.axis([0.0, 1.0, 0.0, 1.0])
    mpl.savefig('MB_FIG12.png', dpi = 600)
    mpl.close()
    #
    '''
    fig1 = mpl.figure()
    mpl.plot(E1, fw, 'ro', markersize = 5, label = '22.5')
    mpl.plot(E2, fw, 'yo', markersize = 5, label = '45')
    mpl.plot(E3, fw, 'mo', markersize = 5, label = '67.5')
    #mpl.plot(u[:, 1], fu, 'b-', markersize = 5, label = 'fui')
    #mpl.plot(w[:, 1], fw, 'g--', markersize = 5, label = 'fwi')
    mpl.legend(loc = 'upper right')
    mpl.xlabel('$E$')
    mpl.ylabel('$f$')
    #mpl.axis([-3.0, 3.0, 0.0, 0.6])
    mpl.savefig('MB_FIG14_E.png', dpi = 600)
    #mpl.show()
    mpl.close()
    '''

################################################################################
if __name__ == '__main__':
    #
    main()
