'''
Name:              Distribution_Cloud.py
Description:       Calculate and Fit Distribution Clouds
Auther:            KhalilWong
License:           MIT License
Copyright:         Copyright (C) 2020, by KhalilWong
Version:           1.5.1
Date:              2020/06/11
Namespace:         https://github.com/KhalilWong/Kernel_MD
DownloadURL:       https://github.com/KhalilWong/Kernel_MD/V_4th_Condensation/Distribution_Cloud_test_new.py
UpdateURL:         https://github.com/KhalilWong/Kernel_MD/V_4th_Condensation/Distribution_Cloud_test_new.py
'''
################################################################################
import numpy as np
import numba as nb
import math
import matplotlib.pyplot as mpl
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import ML_Analysis

################################################################################
def ReadFile(FileName):
    #
    with open(FileName, 'r') as In:
        Data = In.readlines()
        InData = Data[1:]
        ID = []
        Tt = []
        CN = []
        TmVt = []
        X = []
        Y = []
        Z = []
        VX = []
        VY = []
        VZ = []
        FX = []
        FY = []
        FZ = []
        FVX = []
        FVY = []
        FVZ = []
        Adsorbed = []
        for pdata in InData:
            (id, tt, cn, tmvt, x, y, z, vx, vy, vz, fx, fy, fz, fvx, fvy, fvz, ad) = pdata.split('\t', 16)
            ID.append(int(id))
            Tt.append(int(tt))
            CN.append(int(cn))
            TmVt.append(int(tmvt))
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
            VX.append(float(vx))
            VY.append(float(vy))
            VZ.append(float(vz))
            FX.append(float(fx))
            FY.append(float(fy))
            FZ.append(float(fz))
            FVX.append(float(fvx))
            FVY.append(float(fvy))
            FVZ.append(float(fvz))
            if (ad == 'False\n'):
                Adsorbed.append(0)
            elif (ad == 'True\n'):
                Adsorbed.append(1)
    ID = np.array(ID)
    Tt = np.array(Tt)
    CN = np.array(CN)
    TmVt = np.array(TmVt)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    VX = np.array(VX)
    VY = np.array(VY)
    VZ = np.array(VZ)
    FX = np.array(FX)
    FY = np.array(FY)
    FZ = np.array(FZ)
    FVX = np.array(FVX)
    FVY = np.array(FVY)
    FVZ = np.array(FVZ)
    Adsorbed = np.array(Adsorbed)
    #
    return(ID, Tt, CN, TmVt, X, Y, Z, VX, VY, VZ, FX, FY, FZ, FVX, FVY, FVZ, Adsorbed)

################################################################################
def Random_Sampling(N, SamplesN, ID, VX, VY, VZ, FVX, FVY, FVZ, Adsorbed):
    #
    #SamplesID=random.sample(range(N),SamplesN)
    SamplesID = np.random.choice(N, SamplesN, replace = False)
    SampleID = []
    SampleVX = []
    SampleVY = []
    SampleVZ = []
    SampleFVX = []
    SampleFVY = []
    SampleFVZ = []
    SampleAds = []
    SamplesUAN = 0
    for i in range(SamplesN):
        IndexID = np.where(ID == SamplesID[i])
        SampleID.append(int(ID[IndexID]))
        SampleVX.append(float(VX[IndexID]))
        SampleVY.append(float(VY[IndexID]))
        SampleVZ.append(float(VZ[IndexID]))
        SampleFVX.append(float(FVX[IndexID]))
        SampleFVY.append(float(FVY[IndexID]))
        SampleFVZ.append(float(FVZ[IndexID]))
        SampleAds.append(int(Adsorbed[IndexID]))
        if int(Adsorbed[IndexID]) == 0:
            SamplesUAN += 1
    SampleID = np.array(SampleID)
    SampleVX = np.array(SampleVX)
    SampleVY = np.array(SampleVY)
    SampleVZ = np.array(SampleVZ)
    SampleFVX = np.array(SampleFVX)
    SampleFVY = np.array(SampleFVY)
    SampleFVZ = np.array(SampleFVZ)
    SampleAds = np.array(SampleAds)
    #
    return(SamplesUAN, SampleID, SampleVX, SampleVY, SampleVZ, SampleFVX, SampleFVY, SampleFVZ, SampleAds)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Distribution_Cloud(N, UnA_N, X, Y, Adsorbed, I, J, XL = 0.0, XH = 0.0, YL = 0.0, YH = 0.0):
    #
    if XL != XH:
        XLow = XL
        XHigh = XH
    else:
        XLow = np.min(X)
        XHigh = np.max(X)
    if YL != YH:
        YLow = YL
        YHigh = YH
    else:
        YLow = np.min(Y)
        YHigh = np.max(Y)
    dX = (XHigh - XLow) / I
    dY = (YHigh - YLow) / J
    #
    x = np.zeros((I + 1, 3))                                                    #GridLow, GridCenter, GridHigh
    y = np.zeros((J + 1, 3))
    Rf = np.zeros((I + 1, J + 1))
    for i in range(I + 1):
        x[i, 0] = XLow + (i - 0.5) * dX
        x[i, 1] = (XLow + XHigh + (2 * i - I) * dX) / 2
        x[i, 2] = XHigh - (I - i - 0.5) * dX
    for j in range(J + 1):
        y[j, 0] = YLow + (j - 0.5) * dY
        y[j, 1] = (YLow + YHigh + (2 * j - J) * dY) / 2
        y[j, 2] = YHigh - (J - j - 0.5) * dY
    #
    for j in range(J + 1):
        for i in range(I + 1):
            count = 0
            for n in range(N):
                if Adsorbed[n] != 1 and x[i, 0] <= X[n] < x[i, 2] and y[j, 0] <= Y[n] <y[j, 2]:
                    count += 1
            Rf[i, j] = count / (UnA_N * dX * dY)                                #Probability Density
    #
    return(x, y, Rf)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Distribution_Cloud_FourPhase(N, UnA_N, X0, X1, X2, Y, Adsorbed, I, J, XLt = 0.0, XHt = 0.0, XLn = 0.0, XHn = 0.0, YL = 0.0, YH = 0.0):
    #
    if XLt != XHt:
        XLowt = XLt
        XHight = XHt
    else:
        XLowt = min(np.min(X0), np.min(X1))
        XHight = max(np.max(X0), np.max(X1))
    if XLn != XHn:
        XLown = XLn
        XHighn = XHn
    else:
        XLown = np.min(X3)
        XHighn = np.max(X3)
    if YL != YH:
        YLow = YL
        YHigh = YH
    else:
        YLow = np.min(Y)
        YHigh = np.max(Y)
    dXt = (XHight - XLowt) / I
    dXn = (XHighn - XLown) / I
    dY = (YHigh - YLow) / J
    #
    x0 = np.zeros((I + 1, 3))                                                   #GridLow, GridCenter, GridHigh
    x1 = np.zeros((I + 1, 3))
    x2 = np.zeros((I + 1, 3))
    y = np.zeros((J + 1, 3))
    Rf = np.zeros((I + 1, I + 1, I + 1, J + 1))
    for i in range(I + 1):
        x0[i, 0] = XLowt + (i - 0.5) * dXt
        x0[i, 1] = (XLowt + XHight + (2 * i - I) * dXt) / 2
        x0[i, 2] = XHight - (I - i - 0.5) * dXt
        x1[i, 0] = XLowt + (i - 0.5) * dXt
        x1[i, 1] = (XLowt + XHight + (2 * i - I) * dXt) / 2
        x1[i, 2] = XHight - (I - i - 0.5) * dXt
        x2[i, 0] = XLown + (i - 0.5) * dXn
        x2[i, 1] = (XLown + XHighn + (2 * i - I) * dXn) / 2
        x2[i, 2] = XHighn - (I - i - 0.5) * dXn
    for j in range(J + 1):
        y[j, 0] = YLow + (j - 0.5) * dY
        y[j, 1] = (YLow + YHigh + (2 * j - J) * dY) / 2
        y[j, 2] = YHigh - (J - j - 0.5) * dY
    #
    for n in range(N):
        if Adsorbed[n] != 1:
            for j in range(J + 1):
                if y[j, 0] <= Y[n] < y[j, 2]:
                    for i in range(I + 1):
                        if x0[i, 0] <= X0[n] < x0[i, 2]:
                            for m in range(I + 1):
                                if x1[m, 0] <= X1[n] < x1[m, 2]:
                                    for k in range(I + 1):
                                        if x2[k, 0] <= X2[n] < x2[k, 2]:
                                            Rf[i, m, k, j] += 1.0
                                            break
                                    break
                            break
                    break
    Rf /= UnA_N * dXt * dXt * dXn * dY                                          #Probability Density
    #
    return(x0, x1, x2, y, Rf)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Energy_Distribution_Cloud(N, X, Y, X1, Y1, X2, Y2, Adsorbed, I, J, XL = 0.0, XH = 0.0, YL = 0.0, YH = 0.0):
    #
    if XL != XH:
        XLow = XL
        XHigh = XH
    else:
        XLow = np.min(X)
        XHigh = np.max(X)
    if YL != YH:
        YLow = YL
        YHigh = YH
    else:
        YLow = np.min(Y)
        YHigh = np.max(Y)
    dX = (XHigh - XLow) / I
    dY = (YHigh - YLow) / J
    #
    x = np.zeros((I + 1, 3))
    y = np.zeros((J + 1, 3))
    Rf = np.zeros((I + 1, J + 1))
    EnExcMat = np.zeros((I + 1, J + 1))
    EtExcMat = np.zeros((I + 1, J + 1))
    ETExcMat = np.zeros((I + 1, J + 1))
    ERateInMat = np.zeros((I + 1, J + 1))                                       #cos(in)
    ERateOutMat = np.zeros((I + 1, J + 1))                                      #cos(out)
    ERateExcMat = np.zeros((I + 1, J + 1))                                      #cos(in + out)
    #E2ExcMat=np.zeros((I+1,J+1))
    for i in range(I + 1):
        x[i, 0] = XLow + (i - 0.5) * dX
        x[i, 1] = (XLow + XHigh + (2 * i - I) * dX) / 2
        x[i, 2] = XHigh - (I - i - 0.5) * dX
    for j in range(J + 1):
        y[j, 0] = YLow + (j - 0.5) * dY
        y[j, 1] = (YLow + YHigh + (2 * j - J) * dY) / 2
        y[j, 2] = YHigh - (J - j - 0.5) * dY
    #
    for j in range(J + 1):
        for i in range(I + 1):
            count = 0
            sumE0in = 0.0                                                       #n-direction-in
            sumE1in = 0.0                                                       #t-direction-in
            sumE2in = 0.0                                                       #t-direction-in
            sumE0out = 0.0
            sumE1out = 0.0
            sumE2out = 0.0
            for n in range(N):
                if Adsorbed[n] != 1 and x[i, 0] <= X[n] < x[i, 2] and y[j, 0] <= Y[n] < y[j, 2]:
                    count += 1
                    sumE0in += X[n] ** 2
                    sumE1in += X1[n] ** 2
                    sumE2in += X2[n] ** 2
                    sumE0out += Y[n] ** 2
                    sumE1out += Y1[n] ** 2
                    sumE2out += Y2[n] ** 2
            if count != 0:
                EnExcMat[i, j] = (sumE0out - sumE0in) / count
                EtExcMat[i, j] = (sumE1out + sumE2out - sumE1in - sumE2in) / count
                ETExcMat[i, j] = EnExcMat[i, j] + EtExcMat[i, j]
                ERateInMat[i, j] = math.sqrt(sumE0in / (sumE0in + sumE1in + sumE2in))
                ERateOutMat[i, j] = math.sqrt(sumE0out / (sumE0out + sumE1out + sumE2out))
                ERateExcMat[i, j] = ERateInMat[i, j] * ERateOutMat[i, j] - math.sqrt(1 - ERateInMat[i, j] ** 2) * math.sqrt(1 - ERateOutMat[i, j] ** 2)
            else:
                EnExcMat[i, j] = 0.0
                EtExcMat[i, j] = 0.0
                ETExcMat[i, j] = 0.0
                ERateInMat[i, j] = -1.0
                ERateOutMat[i, j] = -1.0
                ERateExcMat[i, j] = -2.0
    #
    return(x, y, EnExcMat, EtExcMat, ETExcMat, ERateInMat, ERateOutMat, ERateExcMat)

################################################################################
def Cloud_dat(Name1, Name2, I, J, x, y, CloudDat1, CloudName1 = 'f', Type = 'MD', CloudDat0 = np.zeros((1, 1)), CloudName0 = 'MD', Loop = 0):
    #
    if Type == 'MD':
        with open(Name1 + '-' + Name2 + '-' + Type + '.dat', 'w') as Out:
            print('TITLE="Distribution_Cloud"', file = Out)
            print('VARIABLES="' + Name1 + '","' + Name2 + '","' + CloudName1 + '"', file = Out)
            print('ZONE I=' + str(I + 1) + ', J=' + str(J + 1) + ', F=POINT', file = Out)
            for j in range(J + 1):
                for i in range(I + 1):
                    if np.isnan(CloudDat1[i, j]):
                        print(x[i, 1], y[j, 1], 0.0, file = Out)
                    else:
                        print(x[i, 1], y[j, 1], CloudDat1[i, j], file = Out)
    elif Type == 'Sample':
        with open(Name1 + '-' + Name2 + '-' + Type + '.dat', 'w') as Out:
            print('TITLE="Distribution_Cloud"', file = Out)
            print('VARIABLES="' + Name1 + '","' + Name2 + '","' + CloudName1 + '","' + CloudName0 + '"', file = Out)
            print('ZONE I=' + str(I + 1) + ', J=' + str(J + 1) + ', F=POINT', file = Out)
            for j in range(J + 1):
                for i in range(I + 1):
                    print(x[i, 1], y[j, 1], CloudDat1[i, j], CloudDat0[i, j], file = Out)
    elif Type == 'Energy':
        with open(Name1 + '-' + Name2 + '-' + Type + '-' + CloudName1 + '.dat', 'w') as Out:
            print('TITLE="Distribution_Cloud"', file = Out)
            print('VARIABLES="' + Name1 + '","' + Name2 + '","' + CloudName1 + '","' + CloudName0 + '"', file = Out)
            print('ZONE I=' + str(I + 1) + ', J=' + str(J + 1) + ', F=POINT', file = Out)
            for j in range(J + 1):
                for i in range(I + 1):
                    print(x[i, 1], y[j, 1], CloudDat1[i, j], CloudDat0[i, j], file = Out)
    elif Type == 'CLL' or Type == 'AC':
        with open(Name1 + '-' + Name2 + '-' + Type + '_Loop' + str(Loop) + '.dat', 'w') as Out:
            print('TITLE="Distribution_Cloud"', file = Out)
            print('VARIABLES="' + Name1 + '","' + Name2 + '","' + CloudName1 + '","' + CloudName0 + '"', file = Out)
            print('ZONE I=' + str(I + 1) + ', J=' + str(J + 1) + ', F=POINT', file = Out)
            for j in range(J + 1):
                for i in range(I + 1):
                    print(x[i, 1], y[j, 1], CloudDat1[i, j], CloudDat0[i, j], file = Out)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Distribution(N, UnA_N, X1, X2, Adsorbed, I, XL = 0.0, XH = 0.0):
    #
    if XL != XH:
        XLow = XL
        XHigh = XH
    else:
        XLow1 = np.min(X1)
        XLow2 = np.min(X2)
        XLow = min(XLow1, XLow2)
        XHigh1 = np.max(X1)
        XHigh2 = np.max(X2)
        XHigh = max(XHigh1, XHigh2)
    dX = (XHigh - XLow) / I
    #
    x = np.zeros(I + 1)
    f1 = np.zeros(I + 1)                                                        #总入射
    f2 = np.zeros(I + 1)                                                        #总反射
    f3 = np.zeros(I + 1)                                                        #被吸附的入射
    f4 = np.zeros(I + 1)                                                        #被反射的入射
    for i in range(I + 1):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for n in range(N):
            if XLow + (i - 0.5) * dX <= X1[n] < XHigh - (I - i - 0.5) * dX:
                count1 += 1
                if (Adsorbed[n] == 1):
                    count3 += 1
                else:
                    count4 += 1
            if Adsorbed[n] != 1 and XLow + (i - 0.5) * dX <= X2[n] < XHigh - (I - i - 0.5) * dX:
                count2 += 1
        x[i] = (XLow + XHigh + (2 * i - I) * dX) / 2
        f1[i] = count1 / (N * dX)
        f2[i] = count2 / (UnA_N * dX)
        if N - UnA_N != 0:
            f3[i] = count3 / ((N - UnA_N) * dX)
        else:
            f3[i] = 0.0
        f4[i] = count4 / (UnA_N * dX)
    #
    return(x, f1, f2, f3, f4)

################################################################################
@nb.jit(nopython = True, nogil = True)
def GetSpeed(N, u, v, w):
    #
    Speed = np.zeros(N)
    for i in range(N):
        Speed[i] = np.sqrt(u[i] ** 2 + v[i] ** 2 + w[i] ** 2)
    return(Speed)

################################################################################
@nb.jit(nopython = True, nogil = True)
def AP_Distribution_Cloud(N, X, Y, Adsorbed, I, J, XL = 0.0, XH = 0.0, YL = 0.0, YH = 0.0):
    #
    if XL != XH:
        XLow = XL
        XHigh = XH
    else:
        XLow = np.min(X)
        XHigh = np.max(X)
    if YL != YH:
        YLow = YL
        YHigh = YH
    else:
        YLow = np.min(Y)
        YHigh = np.max(Y)
    dX = (XHigh - XLow) / I
    dY = (YHigh - YLow) / J
    #
    x = np.zeros((I + 1, 3))
    y = np.zeros((J + 1, 3))
    AP = np.zeros((I + 1, J + 1))                                               #局部吸附概率
    for i in range(I + 1):
        x[i, 0] = XLow + (i - 0.5) * dX
        x[i, 1] = (XLow + XHigh + (2 * i - I) * dX) / 2
        x[i, 2] = XHigh - (I - i - 0.5) * dX
    for j in range(J + 1):
        y[j, 0] = YLow + (j - 0.5) * dY
        y[j, 1] = (YLow + YHigh + (2 * j - J) * dY) / 2
        y[j, 2] = YHigh - (J - j - 0.5) * dY
    #
    for j in range(J + 1):
        for i in range(I + 1):
            count_all = 0
            count_ad = 0
            for n in range(N):
                if XLow + (i - 0.5) * dX <= X[n] < XHigh - (I - i - 0.5) * dX and YLow + (j - 0.5) * dY <= Y[n] < YHigh - (J - j - 0.5) * dY:
                    count_all += 1
                    if Adsorbed[n] == 1:
                        count_ad += 1
            if count_all != 0:
                AP[i, j] = count_ad / count_all
            else:
                AP[i, j] = np.nan
    #
    return(x, y, AP)

################################################################################
@nb.jit(nopython = True, nogil = True)
def AP_Distribution(N, X, Adsorbed, I, XL = 0.0, XH = 0.0):
    #
    if XL != XH:
        XLow = XL
        XHigh = XH
    else:
        XLow = np.min(X)
        XHigh = np.max(X)
    dX = (XHigh - XLow) / I
    #
    x = np.zeros(I + 1)
    AP = np.zeros((I + 1, 2))                                                   #局部吸附概率，累计吸附概率
    count_global = 0
    count_global_ad = 0
    for i in range(I + 1):
        count_local = 0
        count_local_ad = 0
        for n in range(N):
            if XLow + (i - 0.5) * dX <= X[n] < XHigh - (I - i - 0.5) * dX:
                count_local += 1
                if Adsorbed[n] == 1:
                    count_local_ad += 1
        x[i] = (XLow + XHigh + (2 * i - I) * dX) / 2
        count_global += count_local
        count_global_ad += count_local_ad
        if count_local != 0:
            AP[i, 0] = count_local_ad / count_local
        else:
            AP[i, 0] = np.nan
        if count_global != 0:
            AP[i, 1] = count_global_ad / count_global
        else:
            AP[i, 1] = np.nan
    #
    return(x, AP)

################################################################################
def AP_dat(x, APX, APY, APZ, y, APXYZ, Type = 'local'):
    #
    if Type == 'local':
        N = len(x)
        with open('AP_' + Type + '_Ind.dat','w') as Out:
            print('%s\t%s\t%s\t%s' % ('uvw', 'Pu', 'Pv', 'Pw'), file = Out)
            for i in range(N):
                print('%f\t%f\t%f\t%f' % (x[i], APX[i, 0], APY[i, 0], APZ[2 * i, 0]), file = Out)
        N = len(y)
        with open('AP_' + Type + '_Com.dat','w') as Out:
            print('%s\t%s' % ('Speed', 'Pu'), file = Out)
            for i in range(N):
                print('%f\t%f' % (y[i], APXYZ[i, 0]), file = Out)
    elif Type == 'global':
        N = len(x)
        with open('AP_' + Type + '_Ind.dat','w') as Out:
            print('%s\t%s\t%s\t%s' % ('uvw', 'Pu', 'Pv', 'Pw'), file = Out)
            for i in range(N):
                print('%f\t%f\t%f\t%f' % (x[i], APX[i, 1], APY[i, 1], APZ[2 * i, 1]), file = Out)
        N = len(y)
        with open('AP_' + Type + '_Com.dat','w') as Out:
            print('%s\t%s' % ('Speed', 'Pu'), file = Out)
            for i in range(N):
                print('%f\t%f' % (y[i], APXYZ[i, 1]), file = Out)

################################################################################
def AP_Distribution_Plot(x1, x2, x3, x4, f1, f2, f3, f4, Type = 'local'):
    #fig=mpl.figure()
    lab1 = 'X-Adsorption Probability Distribution'
    lab2 = 'Y-Adsorption Probability Distribution'
    lab3 = 'Z-Adsorption Probability Distribution'
    lab4 = 'Speed-Adsorption Probability Distribution'
    #
    fig0, ax0 = mpl.subplots()
    if Type == 'local':
        ax0.plot(x1, f1[:, 0], 'ro-', markersize = 3, label = lab1)
        ax0.plot(x2, f2[:, 0], 'bv-', markersize = 3, label = lab2)
        ax0.plot(x3, f3[:, 0], 'mD-', markersize = 3, label = lab3)
        ax0.plot(x4, f4[:, 0], 'c*-', markersize = 3, label = lab4)
    elif Type == 'global':
        ax0.plot(x1, f1[:, 1], 'ro-', markersize = 3, label = lab1)
        ax0.plot(x2, f2[:, 1], 'bv-', markersize = 3, label = lab2)
        ax0.plot(x3, f3[:, 1], 'mD-', markersize = 3, label = lab3)
        ax0.plot(x4, f4[:, 1], 'c*-', markersize = 3, label = lab4)
    ax0.legend(loc = 'upper right', fontsize = 'small', frameon = False)
    ax0.set_xlabel('$u, v, w$')
    ax0.set_ylabel('$Adsorption Probability$')
    ax0.set_xlim(-3.0, 5.0)
    ax0.set_ylim(0.0, 1.1)
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax0.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax0.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax0.tick_params(axis = 'both', which = 'major', direction = 'in', length = 5, width = 2)
    ax0.tick_params(axis = 'both', which = 'minor', direction = 'in', length = 3, width = 1)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    #
    ax1 = ax0.twinx()
    ax1.set_ylim(0.0, 1.1)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.tick_params(axis = 'y', which = 'major', direction = 'in', length = 5, width = 2)
    ax1.tick_params(axis = 'y', which = 'minor', direction = 'in', length = 3, width = 1)
    ax1.spines['right'].set_linewidth(2)
    #
    ax2 = ax0.twiny()
    ax2.set_xlim(-3.0, 5.0)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.tick_params(axis = 'x', which = 'major', direction = 'in', length = 5, width = 2)
    ax2.tick_params(axis = 'x', which = 'minor', direction = 'in', length = 3, width = 1)
    ax2.spines['top'].set_linewidth(2)
    #
    mpl.savefig('AP_' + Type + '_uvw.png', dpi = 600)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Criterion(N, UnA_N, X, Adsorbed, I, XSN = 10, XL = 0.0, XH = 0.0):
    #
    if XL != XH:
        XLow = XL
        XHigh = XH
    else:
        XLow = np.min(X)
        XHigh = np.max(X)
    dX = (XHigh - XLow) / (I * XSN)
    #
    x = np.zeros((XSN, I + 1))
    TimeCount = np.zeros((XSN, I + 1))
    CriterionTime = np.zeros((XSN, I + 1))
    #CriterionCollision = np.zeros(I + 1)
    for j in range(XSN):
        #
        for i in range(I + 1):
            count = 0
            #count2 = 0
            for n in range(N):
                if Adsorbed[n] != 1 and XLow + (j * I + i - 0.5) * dX <= X[n] < XHigh - ((XSN - j) * I - i - 0.5) * dX:
                    count += 1
            x[j, i] = (XLow + XHigh + (2 * i - (XSN - 2 * j) * I) * dX) / 2
            TimeCount[j, i] = count
            CriterionTime[j, i] = count / (UnA_N * dX)
    #
    return(x, TimeCount, CriterionTime)

################################################################################
def Distribution_Plot(Name, x, f1, f2, f3, f4, f5, f6, f7, XL, XH, YL, YH, GasT):
    fig = mpl.figure()
    if Name == 'VZ':
        lab1 = 'Incident Velocity Distribution with T=' + str(int((abs(x[np.where(f1 == np.max(f1))]) * np.sqrt(2)) ** 2 * GasT)) + ' K'
        lab2 = 'Reflection Velocity Distribution with T=' + str(int((abs(x[np.where(f2 == np.max(f2))]) * np.sqrt(2)) ** 2 * GasT)) + ' K'
        if np.max(f3) != 0.0:
            lab3 = 'Incident Velocity-Adsorbed Distribution with T=' + str(int((abs(x[np.where(f3 == np.max(f3))[0][0]]) * np.sqrt(2)) ** 2 * GasT)) + ' K'
        else:
            lab3 = 'Incident Velocity-Adsorbed Distribution'
        lab4 = 'Incident Velocity-UnAdsorbed Distribution with T=' + str(int((abs(x[np.where(f4 == np.max(f4))]) * np.sqrt(2)) ** 2 * GasT)) + ' K'
        #lab5='CLL Incident Velocity Distribution with T='+str(int((abs(x[np.where(f5==np.max(f5))])*np.sqrt(2))**2*GasT))+' K'
        #lab6='CLL Reflection Velocity Distribution with T='+str(int((abs(x[np.where(f6==np.max(f6))[0][0]+len(f6)-1])*np.sqrt(2))**2*GasT))+' K'
        #x5=x[:len(f5)]
        #x6=x[len(f6)-1:]
        #x7=x[:len(f7)]
    else:
        lab1 = 'Incident Velocity Distribution with T=' + str(int((1 / (np.max(f1) * np.sqrt(np.pi))) ** 2 * GasT)) + ' K'
        lab2 = 'Reflection Velocity Distribution with T=' + str(int((1 / (np.max(f2) * np.sqrt(np.pi))) ** 2 * GasT)) + ' K'
        lab3 = 'Incident Velocity-Adsorbed Distribution with T=' + str(int((1 / (np.max(f3) * np.sqrt(np.pi))) ** 2 * GasT)) + ' K'
        lab4 = 'Incident Velocity-UnAdsorbed Distribution with T=' + str(int((1 / (np.max(f4) * np.sqrt(np.pi))) ** 2 * GasT)) + ' K'
        #lab5='CLL Incident Velocity Distribution with T='+str(int((1/(np.max(f5)*np.sqrt(np.pi)))**2*GasT))+' K'
        #lab6='CLL Reflection Velocity Distribution with T='+str(int((1/(np.max(f6)*np.sqrt(np.pi)))**2*GasT))+' K'
        #x5=x
        #x6=x
        #x7=x
    mpl.plot(x, f1, 'ro', markersize = 4, label = lab1)
    mpl.plot(x, f2, 'go', markersize = 4, label = lab2)
    mpl.plot(x, f3, 'bo', markersize = 4, label = lab3)
    mpl.plot(x, f4, 'mo', markersize = 4, label = lab4)
    #mpl.plot(x5,f5,'r--',markersize=4,label=lab5)
    #mpl.plot(x6,f6,'g--',markersize=4,label=lab6)
    #mpl.plot(x7,f7,'m--',markersize=4,label='Nor')
    mpl.legend(loc = 'upper right', fontsize = 'xx-small')
    mpl.xlabel('$' + str(Name) + '$')
    mpl.ylabel('$f$')
    mpl.axis([XL, XH, YL, YH])
    mpl.savefig('IRA_f_' + Name + '.png', dpi = 600)
    #
    #mpl.show()
    mpl.close()

################################################################################
def FIG12_Distribution_Plot(x, f1, f2, f3, f4, f5, y, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15):
    #fig=mpl.figure()
    lab1 = 'MD UnAdsorbed-Incident Velocity Distribution'
    lab2 = 'MD Reflection Velocity Distribution'
    lab3 = 'CLL Incident Velocity Distribution'
    lab4 = 'CLL Reflection Velocity Distribution'
    #lab5='Maxwellian Distribution'
    lab6 = 'Normalization Condition Distribution'
    x3 = x[:len(f3)]
    x4 = x[len(f4) - 1:]
    x5 = x[:len(f5)]
    #
    fig4, ax4 = mpl.subplots()
    #ax4.set_title('b',loc='left')
    ax4.text(0.05, 0.95, '(b)', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 12)
    ax4.plot(x, f1, 'ro', markersize = 3, label = 'Z: ' + lab1)
    ax4.plot(x, f2, 'bv', markersize = 3, label = 'Z: ' + lab2)
    ax4.plot(x3, f3, 'r', markersize = 3, label = 'Z: ' + lab3)
    ax4.plot(x4, f4, 'b', markersize = 3, label = 'Z: ' + lab4)
    ax4.plot(x5, f5, 'mD', markersize = 3, label = 'Z: ' + lab6)
    #ax4.plot(MX3,MY3,'m',markersize=3,label='Maxwellian')
    ax4.legend(loc = 'upper right', fontsize = 'xx-small', frameon = False)
    ax4.set_xlabel('$w$')
    ax4.set_ylabel('$f$')
    ax4.set_xlim(-3.0, 3.0)
    ax4.set_ylim(0.0, 1.1)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
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
    ax5.set_ylim(0.0, 1.1)
    ax5.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax5.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax5.yaxis.set_major_formatter(ticker.NullFormatter())
    ax5.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax5.tick_params(axis = 'y', which = 'major', direction = 'in', length = 5, width = 2)
    ax5.tick_params(axis = 'y', which = 'minor', direction = 'in', length = 3, width = 1)
    ax5.spines['right'].set_linewidth(2)
    #
    ax6 = ax4.twiny()
    ax6.set_xlim(-3.0, 3.0)
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax6.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax6.xaxis.set_major_formatter(ticker.NullFormatter())
    ax6.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax6.tick_params(axis = 'x', which = 'major', direction = 'in', length = 5, width = 2)
    ax6.tick_params(axis = 'x', which = 'minor', direction = 'in', length = 3, width = 1)
    ax6.spines['top'].set_linewidth(2)
    #
    mpl.savefig('FIG12b_f_vnir.eps', dpi = 600)
    #
    fig1,ax1 = mpl.subplots()
    #ax1.set_title('a',loc='left')
    ax1.text(0.05, 0.95, '(a)', horizontalalignment = 'center', verticalalignment = 'center', transform = ax1.transAxes, fontsize = 12)
    ax1.set_xlabel('$u$')
    ax1.set_xlim(-3.0, 3.0)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    ax1.set_ylabel('$f$')
    ax1.set_ylim(0.0, 1.1)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    #l1=ax1.plot(x1,f1,'ro',markersize=3,label='MD-$u_{r}$')
    l1 = ax1.plot(y, f6, 'ro', markersize = 3, label = 'X: ' + lab1)
    l2 = ax1.plot(y, f7, 'bv', markersize = 3, label = 'X: ' + lab2)
    l3 = ax1.plot(y, f8, 'r', markersize = 3, label = 'X: ' + lab3)
    l4 = ax1.plot(y, f9, 'b', markersize = 3, label = 'X: ' + lab4)
    l5 = ax1.plot(y, f10, 'mD', markersize = 3, label = 'X: ' + lab6)
    #l3=ax1.plot(MX1,MY1,'m',markersize=3,label='Maxwellian')
    ax1.tick_params(axis = 'both', which = 'major', direction = 'in', length = 5, width = 2)
    ax1.tick_params(axis = 'both', which = 'minor', direction = 'in', length = 3, width = 1)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    #
    ax2 = ax1.twiny()
    ax2.set_xlabel('$v$')
    ax2.set_xlim(-3.0, 3.0)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    l6 = ax2.plot(y, f11, 'g^', markersize = 3, label = 'Y: ' + lab1)
    l7 = ax2.plot(y, f12, 'y*', markersize = 3, label = 'Y: ' + lab2)
    l8 = ax2.plot(y, f13, 'g', markersize = 3, label = 'Y: ' + lab3)
    l9 = ax2.plot(y, f14, 'y', markersize = 3, label = 'Y: ' + lab4)
    l10 = ax2.plot(y, f15, 'cD', markersize = 3, label = 'Y: ' + lab6)
    ax2.tick_params(axis = 'x', which = 'major', direction = 'in', length = 5, width = 2)
    ax2.tick_params(axis = 'x', which = 'minor', direction = 'in', length = 3, width = 1)
    ax2.spines['top'].set_linewidth(2)
    #
    ax3 = ax1.twinx()
    ax3.set_ylim(0.0, 1.1)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax3.yaxis.set_major_formatter(ticker.NullFormatter())
    ax3.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.tick_params(axis = 'y', which = 'major', direction = 'in', length = 5, width = 2)
    ax3.tick_params(axis = 'y', which = 'minor', direction = 'in', length = 3, width = 1)
    ax3.spines['right'].set_linewidth(2)
    #
    ls = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10
    legs = [l.get_label() for l in ls]
    ax2.legend(ls, legs, loc = 'upper right', fontsize = 'xx-small', frameon = False)#medium
    #
    mpl.savefig('FIG12a_f_vtir.eps', dpi = 600)
    #
    #mpl.show()
    mpl.close()

################################################################################
def FIG8_Distribution_Plot(mpVelocity, x, f1, f2, f3):
    N = len(x)
    x1 = []
    for i in range(N):
        if i % 2 == 0:
            x1.append(x[i])
    #
    Axis = 2
    MY1 = [np.sqrt(Axis / (2 * np.pi)) * np.exp(- Axis * xi ** 2 / (2)) for xi in x]
    MY3 = [abs(Axis * xi * np.exp(- Axis * xi ** 2 / (2))) for xi in x]
    #
    fig1, ax1 = mpl.subplots()
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
    l1 = ax1.plot(x1, f1, 'ro', markersize = 3, label = 'MD-$u_{r}$')
    l3 = ax1.plot(x, MY1, 'm', markersize = 3, label = 'Maxwellian')
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
    l2 = ax2.plot(x1, f2, 'bv', markersize = 3, label = 'MD-$v_{r}$')
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
    ax1.legend(ls, legs, loc = 'upper right', fontsize = 'medium', frameon = False)#medium
    #
    mpl.savefig('FIG8_f_rvt.eps', dpi = 600)
    #
    fig2, ax4 = mpl.subplots()
    #ax4.set_title('b',loc='left')
    ax4.text(0.05, 0.95, '(b)', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 12)
    ax4.plot(x, f3, 'go', markersize = 3, label = 'MD-$w_{r}$')
    ax4.plot(x, MY3, 'm', markersize = 3, label = 'Maxwellian')
    ax4.legend(loc = 'upper right', fontsize = 'medium', frameon = False)
    ax4.set_xlabel('$w$')
    ax4.set_ylabel('$f$')
    ax4.set_xlim(0.0, 3.0)
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
    ax6.set_xlim(0.0, 3.0)
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax6.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax6.xaxis.set_major_formatter(ticker.NullFormatter())
    ax6.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax6.tick_params(axis = 'x', which = 'major', direction = 'in', length = 5, width = 2)
    ax6.tick_params(axis = 'x', which = 'minor', direction = 'in', length = 3, width = 1)
    ax6.spines['top'].set_linewidth(2)
    #
    mpl.savefig('FIG8_f_rvn.eps', dpi = 600)
    #mpl.show()
    mpl.close()

################################################################################
def Criterion_Plot(Name, x, f1, f2, XSN, Y1L, Y1H, Y2L, Y2H):
    for j in range(XSN):
        xl = int(x[j, 0])
        xh = int(x[j, -1])
        y1l = Y1L
        y1h = min(Y1H, 1.25 * max(f1[j, :]) - 0.25 * y1l)
        y2l = Y2L
        y2h = min(Y2H, 1.25 * max(f2[j, :]) - 0.25 * y2l)
        #
        fig, ax1 = mpl.subplots()
        lab1 = 'UnAdsorbed-IntTime f Distribution'
        ax1.set_xlabel('$' + str(Name) + '$')
        ax1.set_xlim(xl, xh)
        ax1.set_ylabel('$f$', color = 'r')
        ax1.set_ylim(y1l, y1h)
        l1 = ax1.plot(x[j, :], f1[j, :], 'ro', markersize = 4, label = lab1)
        ax1.tick_params(axis = 'y', labelcolor = 'r')
        #
        ax2 = ax1.twinx()
        lab2 = 'UnAdsorbed-IntTime CountN Distribution'
        ax2.set_ylabel('$CountN$', color = 'b')
        ax2.set_ylim(y2l, y2h)
        l2 = ax2.plot(x[j, :], f2[j, :], 'b--', markersize = 4, label = lab2)
        ax2.tick_params(axis = 'y', labelcolor = 'b')
        #
        ls = l1 + l2
        legs = [l.get_label() for l in ls]
        ax1.legend(ls, legs, loc = 'upper right', fontsize = 'x-small')
        mpl.savefig('Criterion_' + Name + '_Segment_' + str(j) + '.png', dpi = 600)
        #
        #mpl.show()
        mpl.close()

################################################################################
@nb.jit(nopython = True, nogil = True)
def CLL_R(Type, ui, vi, wi, ur, l):
    if Type == 0:
        ##sigma = l[0] * np.exp(- l[1] * vi ** 2 - l[2] * wi ** 2)
        sigma1 = l[0]
        sigma2 = l[1]
        sigma3 = l[2]
        #sigma = max(sigma, 1e-32)
        #tt = 1 - math.log(l1 + l2 * vi ** 2 - l3 * wi ** 2)
        #R = 1 / np.sqrt(np.pi * sigma * (2 - sigma)) * np.exp(- (ur - (1 - sigma) * ui) ** 2 / (sigma * (2 - sigma)))
        ##R = 1 / np.sqrt(np.pi * l[3] * sigma * (2 - sigma)) * np.exp(- (ur - (1 - sigma) * ui) ** 2 / (l[3] * sigma * (2 - sigma)))
        #R = 1 / np.sqrt(np.pi * lui * sigma * (2 - sigma)) * np.exp(- (ur - (1 - sigma) * ui) ** 2 / (lui * sigma * (2 - sigma)))
        Rui = 1 / np.sqrt(np.pi * sigma1 * (2 - sigma1)) * np.exp(- (ur - (1 - sigma1) * ui) ** 2 / (sigma1 * (2 - sigma1)))
        Rvi = 1 / np.sqrt(np.pi * sigma2 * (2 - sigma2)) * np.exp(- (ur - (1 - sigma2) * vi) ** 2 / (sigma2 * (2 - sigma2)))
        Rwi = 1 / np.sqrt(np.pi * sigma3 * (2 - sigma3)) * np.exp(- (ur - (1 - sigma3) * wi) ** 2 / (sigma3 * (2 - sigma3)))
        R = l[3] * Rui + l[4] * Rvi + l[5] * Rwi
        #
        '''
        sigma1 = l1
        sigma2 = l2
        #sigma2 = 2.0 - l1
        alpha3 = l3
        R1 = 1 / math.sqrt(np.pi * sigma1 * (2 - sigma1)) * math.exp(- (ur - (1 - sigma1) * ui) ** 2 / (sigma1 * (2 - sigma1)))
        R2 = 1 / math.sqrt(np.pi * sigma2 * (2 - sigma2)) * math.exp(- (ur - (1 - sigma2) * ui) ** 2 / (sigma2 * (2 - sigma2)))
        #R3 = 1 / math.sqrt(np.pi) * math.exp(- (ur ** 2 + alpha3 * ui ** 2))
        R3 = 1 / math.sqrt(np.pi) * math.exp(- ur ** 2) * (1 - min(int(ui ** 2 / alpha3), 1))
        R = l4 * R1 + l5 * R2 + l6 * R3
        #R = l4 * R1 + l4 * R2 + l6 * R3
        #R = l4 * R1 + l5 * R2
        #R = l4 * R1 + l4 * R2
        '''
    elif Type == 1:
        #tt=l1-l2*math.exp(-l3*abs(ur**2-ui**2))
        #tt=1-math.log(l1+l1*vi**2+l3*wi**2)
        #tt=l1+l2*np.cos(l3*(vi**2+wi**2)+l4)
        ##alpha = l[0] * np.exp(- l[1] * vi ** 2 - l[1] * wi ** 2)
        alpha1 = l[0]
        alpha2 = l[1]
        alpha3 = l[2]
        #alpha=tt*(2-tt)
        #alpha=tt
        #alpha1 = l1
        #alpha2 = l2
        N_theta = 100                                                           #1000
        R = 0.0
        R1 = 0.0
        R2 = 0.0
        R3 = 0.0
        for i in range(N_theta):
            #R += 2 * ur / alpha * np.exp(- (ur ** 2 + (1 - alpha) * ui ** 2 - 2 * np.sqrt(1 - alpha) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / alpha) / N_theta
            ##R += 2 * ur / (l[2] * alpha) * np.exp(- (ur ** 2 + (1 - alpha) * ui ** 2 - 2 * np.sqrt(1 - alpha) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / (l[2] * alpha)) / N_theta
            #R += 2 * ur / (lui * alpha) * np.exp(- (ur ** 2 + (1 - alpha) * ui ** 2 - 2 * np.sqrt(1 - alpha) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / (lui * alpha)) / N_theta
            #R += 2 * ur ** 1.5 / (l[2] * alpha) * np.exp(- (ur ** 2 + (1 - alpha) * ui ** 2 - 2 * np.sqrt(1 - alpha) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / (l[2] * alpha)) / N_theta
            R1 += 2 * ur / alpha1 * np.exp(- (ur ** 2 + (1 - alpha1) * ui ** 2 - 2 * np.sqrt(1 - alpha1) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / alpha1) / N_theta
            R2 += 2 * ur / alpha2 * np.exp(- (ur ** 2 + (1 - alpha2) * vi ** 2 - 2 * np.sqrt(1 - alpha2) * ur * vi * np.cos((i + 0.5) / N_theta * np.pi)) / alpha2) / N_theta
            R3 += 2 * ur / alpha3 * np.exp(- (ur ** 2 + (1 - alpha3) * wi ** 2 - 2 * np.sqrt(1 - alpha3) * ur * wi * np.cos((i + 0.5) / N_theta * np.pi)) / alpha3) / N_theta
        R = l[3] * R1 + l[4] * R2 + l[5] * R3
        #
        '''
        alpha1 = l1
        alpha2 = l2
        alpha3 = l3
        N_theta=1000
        #d_theta=np.pi/N_theta
        #R=0.0
        R1 = 0.0
        R2 = 0.0
        R3 = 0.0
        for i in range(N_theta):
            #R+=2*ur/alpha*math.exp(-(ur**2+(1-alpha)*ui**2-2*math.sqrt(1-alpha)*ur*ui*math.cos((i+0.5)/N_theta*np.pi))/alpha)/N_theta
            R1 += 2 * ur / alpha1 * math.exp(- (ur ** 2 + (1 - alpha1) * ui ** 2 - 2 * math.sqrt(1 - alpha1) * ur * ui * math.cos((i + 0.5) / N_theta * np.pi)) / alpha1) / N_theta
            R2 += 2 * ur / alpha2 * math.exp(- (ur ** 2 + (1 - alpha2) * ui ** 2 - 2 * math.sqrt(1 - alpha2) * ur * ui * math.cos((i + 0.5) / N_theta * np.pi)) / alpha2) / N_theta
            #R3 += 2 * ur * math.exp(- (ur ** 2 + alpha3 * ui ** 2)) / N_theta
            R3 += 2 * ur * math.exp(- ur ** 2) * (1 - min(int(ui ** 2 / alpha3), 1)) / N_theta
        #R = l4 * R1 + l5 * R2
        R = l4 * R1 + l5 * R2 + l6 * R3
        '''
    return(R)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Reciprocal(Type, ui, ur, l1, l2, l3, l4):
    #0<l1<1;0<l2<1;l3>0;l4>0
    alpha1 = l1
    alpha2 = l2
    if Type == 0:
        R1 = 1 / np.sqrt(np.pi * alpha1) * np.exp(- (ur - np.sqrt(1 - alpha1) * ui) ** 2 / alpha1)
        R2 = 1 / np.sqrt(np.pi * alpha2) * np.exp(- (ur - np.sqrt(1 - alpha2) * ui) ** 2 / alpha2)
        R = 1 / (l3 / R1 + l4 / R2)
    elif Type == 1:
        N_theta = 1000
        #d_theta=np.pi/N_theta
        R1 = 0.0
        R2 = 0.0
        for i in range(N_theta):
            R1 += 2 * ur / alpha1 * np.exp(- (ur ** 2 + (1 - alpha1) * ui ** 2 - 2 * np.sqrt(1 - alpha1) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / alpha1) / N_theta
            R2 += 2 * ur / alpha2 * np.exp(- (ur ** 2 + (1 - alpha2) * ui ** 2 - 2 * np.sqrt(1 - alpha2) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / alpha2) / N_theta
        R = 1 / (l3 / R1 + l4 / R2)
    return(R)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Epstein(Type, ui, ur, l1, l2, l3, l4):
    #l1>0;0<l2<l1;0<l3<1
    P = np.exp(- l1 * ui ** 2) + l3 * (1 - np.exp(- l2 * ui ** 2))
    a = np.exp(- l4)
    if Type == 0:
        x = ui - ur
        LimDelta = 1 / (a * np.sqrt(np.pi)) * np.exp(- x ** 2 / a ** 2)
        R = P / np.sqrt(np.pi) * np.exp(- ur ** 2) + (1 - P) * LimDelta
    elif Type == 1:
        x = ui + ur
        LimDelta = 1 / (a * np.sqrt(np.pi)) * np.exp(- x ** 2 / a ** 2)
        R=P * 2 * ur * np.exp(- ur ** 2) + (1 - P) * LimDelta
    return(R)
'''
################################################################################
@nb.jit(nopython=True,nogil=True)
def Gradient_Least_Squares_old(Model,Type,N,ui,vi,wi,ur,Rfuir,fui,fvi,fwi,VX,FVX,VY,VZ,Adsorbed,GasTmpV,WallTmpV,lt1,lt2,lt3,lt4,lt5,lt6,mu,v1,v2,v3,v4,v5,v6):
    #
    Mod_Rfu=np.zeros((len(ui),len(ur)))
    Mod_Rfu1=np.zeros((len(ui),len(ur)))
    Mod_Rfu2=np.zeros((len(ui),len(ur)))
    Mod_Rfu3=np.zeros((len(ui),len(ur)))
    Mod_Rfu4=np.zeros((len(ui),len(ur)))
    Mod_Rfu5=np.zeros((len(ui),len(ur)))
    Mod_Rfu6=np.zeros((len(ui),len(ur)))
    Mod_fui=np.zeros(len(ui))
    Mod_fur=np.zeros(len(ur))
    Mod_Nor=np.zeros(len(ui))
    Mod_Fu=0.0
    Err=0.0
    Err1=0.0
    Err2=0.0
    Err3=0.0
    Err4=0.0
    Err5=0.0
    Err6=0.0
    alpha=1e-4
    dlt=1e-8
    #
    for i in range(len(ui)):
        if(fui[i]!=0):
            dfui=fui[i]
            for j in range(len(ur)):
                CountNij=0.0
                for n in range(N):
                    if(Adsorbed[n]!=1 and ui[i,0]<=VX[n]<ui[i,2] and ur[j,0]<=FVX[n]<ur[j,2]):
                        for k in range(len(vi)):
                            if(vi[k,0]<=VY[n]<vi[k,2]):
                                dfvi=fvi[k]
                                break
                        for m in range(len(wi)):
                            if(wi[m,0]<=VZ[n]<wi[m,2]):
                                dfwi=fwi[m]
                                break
                        CountNij+=dfvi*dfwi
                        Mod_Rfu[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,VY[n]*GasTmpV/WallTmpV,VZ[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4,lt5,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu1[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,VY[n]*GasTmpV/WallTmpV,VZ[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1+dlt,lt2,lt3,lt4,lt5,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu2[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,VY[n]*GasTmpV/WallTmpV,VZ[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2+dlt,lt3,lt4,lt5,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu3[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,VY[n]*GasTmpV/WallTmpV,VZ[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3+dlt,lt4,lt5,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu4[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,VY[n]*GasTmpV/WallTmpV,VZ[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4+dlt,lt5,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu5[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,VY[n]*GasTmpV/WallTmpV,VZ[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4,lt5+dlt,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        Mod_Rfu6[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,VY[n]*GasTmpV/WallTmpV,VZ[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4,lt5,lt6+dlt)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        #Mod_Rfu5[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4,lt5+dlt,lt6)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                        #Mod_Rfu6[i,j]+=Model(Type,VX[n]*GasTmpV/WallTmpV,FVX[n]*GasTmpV/WallTmpV,lt1,lt2,lt3,lt4,lt5,lt6+dlt)*GasTmpV/WallTmpV*dfui*dfvi*dfwi
                if(CountNij!=0):
                    Mod_Rfu[i,j]/=CountNij
                    Mod_Rfu1[i,j]/=CountNij
                    Mod_Rfu2[i,j]/=CountNij
                    Mod_Rfu3[i,j]/=CountNij
                    Mod_Rfu4[i,j]/=CountNij
                    Mod_Rfu5[i,j]/=CountNij
                    Mod_Rfu6[i,j]/=CountNij
                Mod_fui[i]+=Mod_Rfu[i,j]*(ur[j,2]-ur[j,0])
                Mod_Nor[i]+=Mod_Rfu[i,j]/dfui*(ur[j,2]-ur[j,0])
                Mod_fur[j]+=Mod_Rfu[i,j]*(ui[i,2]-ui[i,0])
                Mod_Fu+=Mod_Rfu[i,j]*(ui[i,2]-ui[i,0])*(ur[j,2]-ur[j,0])
                Err+=(Rfuir[i,j]-Mod_Rfu[i,j])**2
                Err1+=(Rfuir[i,j]-Mod_Rfu1[i,j])**2
                Err2+=(Rfuir[i,j]-Mod_Rfu2[i,j])**2
                Err3+=(Rfuir[i,j]-Mod_Rfu3[i,j])**2
                Err4+=(Rfuir[i,j]-Mod_Rfu4[i,j])**2
                Err5+=(Rfuir[i,j]-Mod_Rfu5[i,j])**2
                Err6+=(Rfuir[i,j]-Mod_Rfu6[i,j])**2
    dErr_dlt1=(Err1-Err)/dlt
    dErr_dlt2=(Err2-Err)/dlt
    dErr_dlt3=(Err3-Err)/dlt
    dErr_dlt4=(Err4-Err)/dlt
    dErr_dlt5=(Err5-Err)/dlt
    dErr_dlt6=(Err6-Err)/dlt
    #
    pre_v1=v1
    v1=mu*v1
    v1+=-alpha*dErr_dlt1
    lt1+=v1+mu*(v1-pre_v1)
    if(Type==0):
        if(lt1<=0.0):
            lt1=dlt
            v1=-v1
        elif(lt1>=2.0-dlt):
            lt1=2.0-2*dlt
            v1=-v1
    else:
        if(lt1<=0.0):
            lt1=dlt
            v1=-v1
        elif(lt1>=1.0-dlt):
            lt1=1.0-2*dlt
            v1=-v1
    #
    pre_v2=v2
    v2=mu*v2
    v2+=-alpha*dErr_dlt2
    lt2+=v2+mu*(v2-pre_v2)
    if(Type==0):
        if(lt2<=0.0):
            lt2=dlt
            v2=-v2
        elif(lt2>=2.0-dlt):
            lt2=2.0-2*dlt
            v2=-v2
    else:
        if(lt2<=0.0):
            lt2=dlt
            v2=-v2
        elif(lt2>=1.0-dlt):
            lt2=1.0-2*dlt
            v2=-v2
    #
    pre_v3=v3
    v3=mu*v3
    v3+=-alpha*dErr_dlt3
    lt3+=v3+mu*(v3-pre_v3)
    if(Type==0):
        if(lt3<=0.0):
            lt3=dlt
            v3=-v3
        elif(lt3>=3.0-dlt):
            lt3=3.0-2*dlt
            v3=-v3
    else:
        if(lt3<=0.0):
            lt3=dlt
            v3=-v3
        elif(lt3>=3.0-dlt):
            lt3=3.0-2*dlt
            v3=-v3
    #
    pre_v4=v4
    v4=mu*v4
    v4+=-alpha*dErr_dlt4
    lt4+=v4+mu*(v4-pre_v4)
    if(Type==0):
        if(lt4<0.0):
            lt4=dlt
            v4=-v4
    else:
        if(lt4<0.0):
            lt4=dlt
            v4=-v4
    #
    pre_v5=v5
    v5=mu*v5
    v5+=-alpha*dErr_dlt5
    lt5+=v5+mu*(v5-pre_v5)
    if(Type==0):
        if(lt5<0.0):
            lt5=dlt
            v5=-v5
    else:
        if(lt5<0.0):
            lt5=dlt
            v5=-v5
    #
    pre_v6=v6
    v6=mu*v6
    v6+=-alpha*dErr_dlt6
    lt6+=v6+mu*(v6-pre_v6)
    if(Type==0):
        if(lt6<0.0):
            lt6=dlt
            v6=-v6
    else:
        if(lt6<0.0):
            lt6=dlt
            v6=-v6
    #
    return(Mod_Rfu,Mod_Nor,Mod_Fu,Mod_fui,Mod_fur,Err,lt1,lt2,lt3,lt4,lt5,lt6,v1,v2,v3,v4,v5,v6)
'''

################################################################################
@nb.jit(nopython = True, nogil = True)
def Gradient_Least_Squares(Model, Type, ui, vi, wi, ur, Rfuir, fui, fvi, fwi, GasTmpV, WallTmpV, mu, lt, v):
    #
    Mod_Rfu = np.zeros((len(ui), len(ur)))
    Mod_Rfu_dlt = np.zeros((len(ui), len(ur), len(lt)))
    Mod_fui = np.zeros(len(ui))
    Mod_fur = np.zeros(len(ur))
    Mod_Nor = np.zeros(len(ui))
    Mod_Fu = 0.0
    Err = 0.0
    Err_dlt = np.zeros(len(lt))
    alpha = 1e-4
    dlt = 1e-8
    Lim_lt_t = np.ones((len(lt), 2))
    #Lim_lt_t[:, 0] *= -100.0
    Lim_lt_t[:, 0] *= 0.0
    #Lim_lt_t[:, 1] *= 100.0
    Lim_lt_t[:, 1] *= 1.0
    #Lim_lt_t = np.array([[-100.0, 100.0] for p in range(len(lt))])
    #Lim_lt_t[0, 0] = 0.0
    #Lim_lt_t[0, 1] = 2.0
    #Lim_lt_t[1, 0] = 0.0
    #Lim_lt_t[2, 0] = 0.0
    #Lim_lt_t[3, 0] = 1.0                                                        #p3下限
    Lim_lt_t[0, 1] = 2.0
    Lim_lt_t[1, 1] = 2.0
    Lim_lt_t[2, 1] = 2.0
    Lim_lt_n = np.ones((len(lt), 2))
    #Lim_lt_n[:, 0] *= -100.0
    Lim_lt_n[:, 0] *= 0.0
    #Lim_lt_n[:, 1] *= 100.0
    Lim_lt_n[:, 1] *= 1.0
    #Lim_lt_n = np.array([[-100.0, 100.0] for p in range(len(lt))])
    #Lim_lt_n[0, 0] = 0.0
    #Lim_lt_n[0, 1] = 1.0
    #Lim_lt_n[1, 0] = 0.0
    #Lim_lt_n[2, 0] = 1.0                                                        #p2下限
    #
    for i in range(len(ui)):
        if fui[i] != 0:
            dfui = fui[i]
            for j in range(len(ur)):
                #
                for k in range(len(vi)):
                    for m in range(len(wi)):
                        Mod_Rfu[i, j] += Model(Type, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, lt) * GasTmpV / WallTmpV * dfui * fvi[k] * (vi[k, 2] - vi[k, 0]) * fwi[m] * (wi[m, 2] - wi[m, 0])
                        for p in range(len(lt)):
                            lt[p] += dlt
                            Mod_Rfu_dlt[i, j, p] += Model(Type, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, lt) * GasTmpV / WallTmpV * dfui * fvi[k] * (vi[k, 2] - vi[k, 0]) * fwi[m] * (wi[m, 2] - wi[m, 0])
                            lt[p] -= dlt
                Mod_fui[i] += Mod_Rfu[i, j] * (ur[j, 2] - ur[j, 0])
                Mod_Nor[i] += Mod_Rfu[i, j] / dfui * (ur[j, 2] - ur[j, 0])
                Mod_fur[j] += Mod_Rfu[i, j] * (ui[i, 2] - ui[i, 0])
                Mod_Fu += Mod_Rfu[i, j] * (ui[i, 2] - ui[i, 0]) * (ur[j, 2] - ur[j, 0])
                Err += (Rfuir[i, j] - Mod_Rfu[i, j]) ** 2
                for p in range(len(lt)):
                    Err_dlt[p] += (Rfuir[i, j] - Mod_Rfu_dlt[i, j, p]) ** 2
    dErr_dlt = np.zeros(len(lt))
    for p in range(len(lt)):
        dErr_dlt[p] = (Err_dlt[p] - Err) / dlt
        pre_v = v[p]
        v[p] *= mu
        v[p] += - alpha * dErr_dlt[p]
        lt[p] += v[p] + mu * (v[p] - pre_v)
        if Type == 0:
            if lt[p] <= Lim_lt_t[p, 0]:
                lt[p] = Lim_lt_t[p, 0] + alpha#dlt
                v[p] = - v[p]
            elif lt[p] >= Lim_lt_t[p, 1] - dlt:
                lt[p] = Lim_lt_t[p, 1] - 2 * alpha#dlt
                v[p] = - v[p]
        elif Type == 1:
            if lt[p] <= Lim_lt_n[p, 0]:
                lt[p] = Lim_lt_n[p, 0] + alpha#dlt
                v[p] = - v[p]
            elif lt[p] >= Lim_lt_n[p, 1] - dlt:
                lt[p] = Lim_lt_n[p, 1] - 2 * alpha#dlt
                v[p] = - v[p]
    #
    return(Mod_Rfu, Mod_Nor, Mod_Fu, Mod_fui, Mod_fur, Err, lt, v)

'''
################################################################################
@nb.jit(nopython = True, nogil = True)
def Gradient_Least_Squares(Model, Type, ui, vi, wi, ur, Rfuir, fui, fvi, fwi, GasTmpV, WallTmpV, mu, lt, v, ltui, vui):
    #
    Mod_Rfu = np.zeros((len(ui), len(ur)))
    Mod_Rfu_dltui = np.zeros((len(ui), len(ur)))
    Mod_Rfu_dlt = np.zeros((len(ui), len(ur), len(lt)))
    Mod_fui = np.zeros(len(ui))
    Mod_fur = np.zeros(len(ur))
    Mod_Nor = np.zeros(len(ui))
    Mod_Fu = 0.0
    Err = 0.0
    Err_dltui0 = np.zeros(len(ui))
    Err_dltui1 = np.zeros(len(ui))
    Err_dlt = np.zeros(len(lt))
    alpha = 1e-4
    dlt = 1e-8
    Lim_lt_t = np.ones((len(lt), 2))
    Lim_lt_t[:, 0] *= -100.0
    Lim_lt_t[:, 1] *= 100.0
    #Lim_lt_t = np.array([[-100.0, 100.0] for p in range(len(lt))])
    Lim_lt_t[0, 0] = 0.0
    Lim_lt_t[0, 1] = 2.0
    Lim_lt_t[1, 0] = 0.0
    Lim_lt_t[2, 0] = 0.0
    Lim_lt_t[3, 0] = 1.0
    Lim_lt_n = np.ones((len(lt), 2))
    Lim_lt_n[:, 0] *= -100.0
    Lim_lt_n[:, 1] *= 100.0
    #Lim_lt_n = np.array([[-100.0, 100.0] for p in range(len(lt))])
    Lim_lt_n[0, 0] = 0.0
    Lim_lt_n[0, 1] = 1.0
    Lim_lt_n[1, 0] = 0.0
    Lim_lt_n[2, 0] = 1.0
    #
    for i in range(len(ui)):
        if fui[i] != 0:
            dfui = fui[i]
            for j in range(len(ur)):
                #
                for k in range(len(vi)):
                    for m in range(len(wi)):
                        Mod_Rfu[i, j] += Model(Type, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, lt, ltui[i]) * GasTmpV / WallTmpV * dfui * fvi[k] * (vi[k, 2] - vi[k, 0]) * fwi[m] * (wi[m, 2] - wi[m, 0])
                        Mod_Rfu_dltui[i, j] += Model(Type, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, lt, ltui[i] + dlt) * GasTmpV / WallTmpV * dfui * fvi[k] * (vi[k, 2] - vi[k, 0]) * fwi[m] * (wi[m, 2] - wi[m, 0])
                        for p in range(len(lt)):
                            lt[p] += dlt
                            Mod_Rfu_dlt[i, j, p] += Model(Type, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, lt, ltui[i]) * GasTmpV / WallTmpV * dfui * fvi[k] * (vi[k, 2] - vi[k, 0]) * fwi[m] * (wi[m, 2] - wi[m, 0])
                            lt[p] -= dlt
                Mod_fui[i] += Mod_Rfu[i, j] * (ur[j, 2] - ur[j, 0])
                Mod_Nor[i] += Mod_Rfu[i, j] / dfui * (ur[j, 2] - ur[j, 0])
                Mod_fur[j] += Mod_Rfu[i, j] * (ui[i, 2] - ui[i, 0])
                Mod_Fu += Mod_Rfu[i, j] * (ui[i, 2] - ui[i, 0]) * (ur[j, 2] - ur[j, 0])
                Err += (Rfuir[i, j] - Mod_Rfu[i, j]) ** 2
                for p in range(len(lt)):
                    Err_dlt[p] += (Rfuir[i, j] - Mod_Rfu_dlt[i, j, p]) ** 2
                Err_dltui0[i] += (Rfuir[i, j] - Mod_Rfu[i, j]) ** 2
                Err_dltui1[i] += (Rfuir[i, j] - Mod_Rfu_dltui[i, j]) ** 2
            dErr_dltui = (Err_dltui1[i] - Err_dltui0[i]) / dlt
            pre_v = vui[i]
            vui[i] *= mu
            vui[i] += - alpha * dErr_dltui
            ltui[i] += vui[i] + mu * (vui[i] - pre_v)
            if Type == 0:
                if ltui[i] <= 1.0:
                    ltui[i] = 1.0 + alpha#dlt
                    vui[i] = - vui[i]
                elif ltui[i] >= 100.0 - dlt:
                    ltui[i] = 100.0 - 2 * alpha#dlt
                    vui[i] = - vui[i]
            elif Type == 1:
                if ltui[i] <= 1.0:
                    ltui[i] = 1.0 + alpha#dlt
                    vui[i] = - vui[i]
                elif ltui[i] >= 100.0 - dlt:
                    ltui[i] = 100.0 - 2 * alpha#dlt
                    vui[i] = - vui[i]
    dErr_dlt = np.zeros(len(lt))
    for p in range(len(lt)):
        dErr_dlt[p] = (Err_dlt[p] - Err) / dlt
        pre_v = v[p]
        v[p] *= mu
        v[p] += - alpha * dErr_dlt[p]
        lt[p] += v[p] + mu * (v[p] - pre_v)
        if Type == 0:
            if lt[p] <= Lim_lt_t[p, 0]:
                lt[p] = Lim_lt_t[p, 0] + alpha#dlt
                v[p] = - v[p]
            elif lt[p] >= Lim_lt_t[p, 1] - dlt:
                lt[p] = Lim_lt_t[p, 1] - 2 * alpha#dlt
                v[p] = - v[p]
        elif Type == 1:
            if lt[p] <= Lim_lt_n[p, 0]:
                lt[p] = Lim_lt_n[p, 0] + alpha#dlt
                v[p] = - v[p]
            elif lt[p] >= Lim_lt_n[p, 1] - dlt:
                lt[p] = Lim_lt_n[p, 1] - 2 * alpha#dlt
                v[p] = - v[p]
    #
    return(Mod_Rfu, Mod_Nor, Mod_Fu, Mod_fui, Mod_fur, Err, lt, v, ltui, vui)
'''
################################################################################
@nb.jit(nopython = True, nogil = True)
def Gradient_Least_Squares_FourPhase(Model, Type, Reg_Type, ui, vi, wi, ur, Rfuir, fui, fvi, fwi, GasTmpV, WallTmpV, mu, lt, v):
    #
    Mod_Rfu = np.zeros((len(ui), len(vi), len(wi), len(ur)))
    Mod_Rfu_dlt = np.zeros((len(ui), len(vi), len(wi), len(ur), len(lt)))
    Mod_Rfuiur = np.zeros((len(ui), len(ur)))
    Mod_fui = np.zeros(len(ui))
    Mod_fur = np.zeros(len(ur))
    Mod_Nor = np.zeros(len(ui))
    Mod_Fu = 0.0
    Err = 0.0
    Err_dlt = np.zeros(len(lt))
    alpha = 1e-4
    dlt = 1e-8
    lam = 1.0
    Lim_lt_t = np.ones((len(lt), 2))
    #Lim_lt_t[:, 0] *= -100.0
    Lim_lt_t[:, 0] *= 0.0
    #Lim_lt_t[:, 1] *= 100.0
    Lim_lt_t[:, 1] *= 1.0
    #Lim_lt_t = np.array([[-100.0, 100.0] for p in range(len(lt))])
    #Lim_lt_t[0, 0] = 0.0
    #Lim_lt_t[0, 1] = 2.0
    #Lim_lt_t[1, 0] = 0.0
    #Lim_lt_t[2, 0] = 0.0
    #Lim_lt_t[3, 0] = 1.0                                                        #p3下限
    Lim_lt_t[0, 1] = 2.0
    Lim_lt_t[1, 1] = 2.0
    Lim_lt_t[2, 1] = 2.0
    Lim_lt_n = np.ones((len(lt), 2))
    #Lim_lt_n[:, 0] *= -100.0
    Lim_lt_n[:, 0] *= 0.0
    #Lim_lt_n[:, 1] *= 100.0
    Lim_lt_n[:, 1] *= 1.0
    #Lim_lt_n = np.array([[-100.0, 100.0] for p in range(len(lt))])
    #Lim_lt_n[0, 0] = 0.0
    #Lim_lt_n[0, 1] = 1.0
    #Lim_lt_n[1, 0] = 0.0
    #Lim_lt_n[2, 0] = 1.0                                                        #p2下限
    #
    for i in range(len(ui)):
        if fui[i] != 0:
            dfui = fui[i]
            for j in range(len(ur)):
                #
                for k in range(len(vi)):
                    for m in range(len(wi)):
                        Mod_Rfu[i, k, m, j] = Model(Type, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, lt) * GasTmpV / WallTmpV * dfui * fvi[k]  * fwi[m]
                        for p in range(len(lt)):
                            lt[p] += dlt
                            Mod_Rfu_dlt[i, k, m, j, p] = Model(Type, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, lt) * GasTmpV / WallTmpV * dfui * fvi[k]  * fwi[m]
                            lt[p] -= dlt
                        Mod_Rfuiur[i, j] += Mod_Rfu[i, k, m, j] * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
                        Mod_fui[i] += Mod_Rfu[i, k, m, j] * (ur[j, 2] - ur[j, 0]) * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
                        Mod_Nor[i] += Mod_Rfu[i, k, m, j] / dfui * (ur[j, 2] - ur[j, 0]) * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
                        Mod_fur[j] += Mod_Rfu[i, k, m, j] * (ui[i, 2] - ui[i, 0]) * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
                        Mod_Fu += Mod_Rfu[i, k, m, j] * (ui[i, 2] - ui[i, 0]) * (ur[j, 2] - ur[j, 0]) * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
                        Err += (Rfuir[i, k, m, j] - Mod_Rfu[i, k, m, j]) ** 2
                        for p in range(len(lt)):
                            Err_dlt[p] += (Rfuir[i, k, w, j] - Mod_Rfu_dlt[i, k, w, j, p]) ** 2
    #
    if Reg_Type == '':
        pass
    elif Reg_Type == 'L1':
        for m in range(len(lt)):
            Err += lam * np.abs(lt[m])
            for j in range(len(lt)):
                if j == m:
                    Err_dlt[j] += lam * np.abs(lt[m] + dlt)
                else:
                    Err_dlt[j] += lam * np.abs(lt[m])
        #print(Error)
    elif Reg_Type == 'L2':
        for m in range(len(lt)):
            Err += 0.5 * lam * lt[m] ** 2
            for j in range(len(lt)):
                if j == m:
                    Err_dlt[j] += 0.5 * lam * (lt[m] + dlt) ** 2
                else:
                    Err_dlt[j] += 0.5 * lam * lt[m] ** 2
        #print(Error)
    #
    dErr_dlt = np.zeros(len(lt))
    for p in range(len(lt)):
        dErr_dlt[p] = (Err_dlt[p] - Err) / dlt
        pre_v = v[p]
        v[p] *= mu
        v[p] += - alpha * dErr_dlt[p]
        lt[p] += v[p] + mu * (v[p] - pre_v)
        if Type == 0:
            if lt[p] <= Lim_lt_t[p, 0]:
                lt[p] = Lim_lt_t[p, 0] + alpha#dlt
                v[p] = - v[p]
            elif lt[p] >= Lim_lt_t[p, 1] - dlt:
                lt[p] = Lim_lt_t[p, 1] - 2 * alpha#dlt
                v[p] = - v[p]
        elif Type == 1:
            if lt[p] <= Lim_lt_n[p, 0]:
                lt[p] = Lim_lt_n[p, 0] + alpha#dlt
                v[p] = - v[p]
            elif lt[p] >= Lim_lt_n[p, 1] - dlt:
                lt[p] = Lim_lt_n[p, 1] - 2 * alpha#dlt
                v[p] = - v[p]
    #
    return(Mod_Rfuiur, Mod_Nor, Mod_Fu, Mod_fui, Mod_fur, Err, lt, v)


################################################################################
@nb.jit(nopython = True, nogil = True)
def Irrelevant(Model, Type, N, ui, vr, vi, wi, fui, fvi, fwi, GasTmpV, WallTmpV, lt):
    #
    Mod_Rfu = np.zeros((len(ui), len(vr)))
    #
    for i in range(len(ui)):
        if fui[i] != 0:
            dfui = fui[i]
            for j in range(len(vr)):                                            #vr, wr
                for k in range(len(vi)):
                    for m in range(len(wi)):
                        if Type == 0:
                            Mod_Rfu[i, j] += Model(Type, vi[k, 1] * GasTmpV / WallTmpV, ui[i, 1] * GasTmpV / WallTmpV, wi[m, 1] * GasTmpV / WallTmpV, vr[j, 1] * GasTmpV / WallTmpV, lt) * GasTmpV / WallTmpV * dfui * fvi[k] * fwi[m] * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
                        elif Type == 1:
                            Mod_Rfu[i, j] += Model(Type, wi[m, 1] * GasTmpV / WallTmpV, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, vr[j, 1] * GasTmpV / WallTmpV, lt) * GasTmpV / WallTmpV * dfui * fvi[k] * fwi[m] * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
    return(Mod_Rfu)

################################################################################
@nb.jit(nopython = True, nogil = True)
def CLL_O(Type, ui, ur, AC):
    if Type == 0:
        R = 1 / np.sqrt(np.pi * AC * (2 - AC)) * np.exp(- (ur - (1 - AC) * ui) ** 2 / (AC * (2 - AC)))
    elif Type == 1:
        N_theta = 1000
        #d_theta=np.pi/N_theta
        R = 0.0
        for i in range(N_theta):
            R += 2 * ur / AC * np.exp(- (ur ** 2 + (1 - AC) * ui ** 2 - 2 * np.sqrt(1 - AC) * ur * ui * np.cos((i + 0.5) / N_theta * np.pi)) / AC) / N_theta
    return(R)

################################################################################
@nb.jit(nopython = True, nogil = True)
def CLL_Gen(Type, ui, ur, params0, params1, params2):
    if Type == 0:
        R = params1 / (params0 * np.sqrt(2 * np.pi)) * np.exp(- (ur - params2 * ui) ** 2 / (2 * params0 ** 2))
    elif Type == 1:
        R = (params1 / params0) ** 2 * ur * np.exp(- (ur + params2 * ui) ** 2 / (2 * params0 ** 2))
    return(R)

################################################################################
@nb.jit(nopython = True, nogil = True)
def AC(Type, ui, ur, Rfuir, fui, GasTmpV, WallTmpV, ACMat, VMat, ACHighLim = 1.0):
    #
    CLL_Rfu = np.zeros((len(ui), len(ur)))
    CLL_Rfu_pdac = np.zeros((len(ui), len(ur)))
    CLL_Rfu_mdac = np.zeros((len(ui), len(ur)))
    Err = 0.0
    alpha = 1e-4
    dlt = 1e-8
    mu = 0.999
    for i in range(len(ui)):
        if fui[i] != 0:
            dfui = fui[i]
            for j in range(len(ur)):
                CLL_Rfu[i, j] = CLL_O(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, ACMat[i, j]) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_pdac[i, j] = CLL_O(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, ACMat[i, j] + dlt) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_mdac[i, j] = CLL_O(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, ACMat[i, j] - dlt) * GasTmpV / WallTmpV * dfui
                Err += (Rfuir[i, j] - CLL_Rfu[i, j]) ** 2
                Errp = (Rfuir[i, j] - CLL_Rfu_pdac[i, j]) ** 2
                Errm = (Rfuir[i, j] - CLL_Rfu_mdac[i, j]) ** 2
                dErr_dlt = (Errp - Errm) / (2 * dlt)
                #
                pre_v = VMat[i, j]
                VMat[i, j] = mu * VMat[i, j]
                VMat[i, j] += - alpha * dErr_dlt
                ACMat[i, j] += VMat[i, j] + mu * (VMat[i, j] - pre_v)
                if ACMat[i, j] <= dlt:
                    ACMat[i, j] = 2 * dlt
                    VMat[i, j] = - VMat[i, j]
                elif ACMat[i, j] >= ACHighLim - dlt:
                    ACMat[i, j] = ACHighLim - 2 * dlt
                    VMat[i, j] = - VMat[i, j]
    #
    return(CLL_Rfu, Err, ACMat, VMat)

################################################################################
@nb.jit(nopython = True, nogil = True)
def AC_Vec(Type, ui, ur, Rfuir, fui, GasTmpV, WallTmpV, PA_iVec, V_iVec):
    #
    CLL_Rfu = np.zeros((len(ui), len(ur)))
    CLL_Rfu_pdac0 = np.zeros((len(ui), len(ur)))
    CLL_Rfu_mdac0 = np.zeros((len(ui), len(ur)))
    CLL_Rfu_pdac1 = np.zeros((len(ui), len(ur)))
    CLL_Rfu_mdac1 = np.zeros((len(ui), len(ur)))
    CLL_Rfu_pdac2 = np.zeros((len(ui), len(ur)))
    CLL_Rfu_mdac2 = np.zeros((len(ui), len(ur)))
    Err = 0.0
    alpha = 1e-4
    dlt = 1e-8
    mu = 0.999
    #if(Type == 0):
    #    ACHighLim = 2.0
    #else:
    #    ACHighLim = 1.0
    for i in range(len(ui)):
        if fui[i] != 0:
            dfui = fui[i]
            Errp0 = 0.0
            Errm0 = 0.0
            Errp1 = 0.0
            Errm1 = 0.0
            Errp2 = 0.0
            Errm2 = 0.0
            for j in range(len(ur)):
                #CLL_Rfu[i, j] = CLL_O(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, AC_iVec[i]) * GasTmpV / WallTmpV * dfui
                #CLL_Rfu_pdac[i, j] = CLL_O(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, AC_iVec[i] + dlt) * GasTmpV / WallTmpV * dfui
                #CLL_Rfu_mdac[i, j] = CLL_O(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, AC_iVec[i] - dlt) * GasTmpV / WallTmpV * dfui
                CLL_Rfu[i, j] = CLL_Gen(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, PA_iVec[i, 0], PA_iVec[i, 1], PA_iVec[i, 2]) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_pdac0[i, j] = CLL_Gen(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, PA_iVec[i, 0] + dlt, PA_iVec[i, 1], PA_iVec[i, 2]) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_mdac0[i, j] = CLL_Gen(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, PA_iVec[i, 0] - dlt, PA_iVec[i, 1], PA_iVec[i, 2]) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_pdac1[i, j] = CLL_Gen(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, PA_iVec[i, 0], PA_iVec[i, 1] + dlt, PA_iVec[i, 2]) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_mdac1[i, j] = CLL_Gen(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, PA_iVec[i, 0], PA_iVec[i, 1] - dlt, PA_iVec[i, 2]) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_pdac2[i, j] = CLL_Gen(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, PA_iVec[i, 0], PA_iVec[i, 1], PA_iVec[i, 2] + dlt) * GasTmpV / WallTmpV * dfui
                CLL_Rfu_mdac2[i, j] = CLL_Gen(Type, ui[i, 1] * GasTmpV / WallTmpV, ur[j, 1] * GasTmpV / WallTmpV, PA_iVec[i, 0], PA_iVec[i, 1], PA_iVec[i, 2] - dlt) * GasTmpV / WallTmpV * dfui
                Err += (Rfuir[i, j] - CLL_Rfu[i, j]) ** 2
                Errp0 += (Rfuir[i, j] - CLL_Rfu_pdac0[i, j]) ** 2
                Errm0 += (Rfuir[i, j] - CLL_Rfu_mdac0[i, j]) ** 2
                Errp1 += (Rfuir[i, j] - CLL_Rfu_pdac1[i, j]) ** 2
                Errm1 += (Rfuir[i, j] - CLL_Rfu_mdac1[i, j]) ** 2
                Errp2 += (Rfuir[i, j] - CLL_Rfu_pdac2[i, j]) ** 2
                Errm2 += (Rfuir[i, j] - CLL_Rfu_mdac2[i, j]) ** 2
            #
            dErr_dlt0 = (Errp0 - Errm0) / (2 * dlt)
            dErr_dlt1 = (Errp1 - Errm1) / (2 * dlt)
            dErr_dlt2 = (Errp2 - Errm2) / (2 * dlt)
            #
            pre_v = V_iVec[i, 0]
            V_iVec[i, 0] = mu * V_iVec[i, 0]
            V_iVec[i, 0] += - alpha * dErr_dlt0
            PA_iVec[i, 0] += V_iVec[i, 0] + mu * (V_iVec[i, 0] - pre_v)
            if PA_iVec[i, 0] <= dlt:
                PA_iVec[i, 0] = 2 * dlt
                V_iVec[i, 0] = - V_iVec[i, 0]
            elif PA_iVec[i, 0] >= 2.0 - dlt:
                PA_iVec[i, 0] = 2.0 - 2 * dlt
                V_iVec[i, 0] = - V_iVec[i, 0]
            #
            pre_v = V_iVec[i, 1]
            V_iVec[i, 1] = mu * V_iVec[i, 1]
            V_iVec[i, 1] += - alpha * dErr_dlt1
            PA_iVec[i, 1] += V_iVec[i, 1] + mu * (V_iVec[i, 1] - pre_v)
            if PA_iVec[i, 1] <= dlt:
                PA_iVec[i, 1] = 2 * dlt
                V_iVec[i, 1] = - V_iVec[i, 1]
            elif PA_iVec[i, 1] >= 2.0 - dlt:
                PA_iVec[i, 1] = 2.0 - 2 * dlt
                V_iVec[i, 1] = - V_iVec[i, 1]
            #
            pre_v = V_iVec[i, 2]
            V_iVec[i, 2] = mu * V_iVec[i, 2]
            V_iVec[i, 2] += - alpha * dErr_dlt2
            PA_iVec[i, 2] += V_iVec[i, 2] + mu * (V_iVec[i, 2] - pre_v)
            if PA_iVec[i, 2] <= dlt:
                PA_iVec[i, 2] = 2 * dlt
                V_iVec[i, 2] = - V_iVec[i, 2]
            elif PA_iVec[i, 2] >= 1.0 - dlt:
                PA_iVec[i, 2] = 1.0 - 2 * dlt
                V_iVec[i, 2] = - V_iVec[i, 2]
    #
    return(CLL_Rfu, Err, PA_iVec, V_iVec)

################################################################################
def Err_Plot(Name, X_l, Y_Err, Y_lt):
    Colors = ['r', 'b', 'g', 'm', 'c', 'y']
    #
    fig1 = mpl.figure()
    mpl.plot(X_l, Y_Err, 'ro', markersize = 4, label = 'Last Error=' + str(Y_Err[-1]))
    mpl.legend(loc = 'upper right', fontsize = 'x-small')
    mpl.xlabel('$Loop$')
    mpl.ylabel('$Error$')
    mpl.savefig(Name + '_Error_Loop.png', dpi = 600)
    #
    fig2 = mpl.figure()
    for p in range(len(Y_lt[0, :])):
        mpl.plot(X_l, Y_lt[:, p], Colors[p] + 'o', markersize = 4, label = 'Last lt' + str(p) + '=' + str(Y_lt[-1, p]))
    mpl.legend(loc = 'upper right', fontsize = 'x-small')
    mpl.xlabel('$Loop$')
    mpl.ylabel('$lt$')
    mpl.savefig(Name + '_lt_Loop.png', dpi = 600)
    #
    #mpl.show()
    mpl.close()

################################################################################
def main():
    FileName = 'Incident_Reflection.data'
    ID, Tt, CN, TmVt, X, Y, Z, VX, VY, VZ, FX, FY, FZ, FVX, FVY, FVZ, Adsorbed = ReadFile(FileName)
    #
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
    #
    N = len(ID)
    for i in range(N):
        Tt[i] = Tt[i] * 0.001
    UnA_N = 0
    Ad_N = 0
    for i in range(len(Adsorbed)):
        if Adsorbed[i] == 0:
            UnA_N += 1
        else:
            Ad_N += 1
    print(UnA_N, Ad_N)
    #
    SamplesN = int(N / 10)
    SamplesUAN, SampleID, SampleVX, SampleVY, SampleVZ, SampleFVX, SampleFVY, SampleFVZ, SampleAds = Random_Sampling(N, SamplesN, ID, VX, VY, VZ, FVX, FVY, FVZ, Adsorbed)
    #
    I = 30#20#50
    #
    #cri_Time, Time_Count, cri_f = Criterion(N, UnA_N, Tt, Adsorbed, 200, 35, 0.0, 3500.0)
    #Criterion_Plot('Interaction Time', cri_Time, cri_f, Time_Count, 35, 0.0, 0.5, 0.0, 7000.0)
    #
    ui, tu, Rfuit = Distribution_Cloud(N, UnA_N, VX, Tt, Adsorbed, I, I, -3.0, 3.0, 0.0, 50.0)
    #Cloud_dat('VX', 'Tt', I, I, ui, tu, Rfuit)
    vi, tv, Rfvit = Distribution_Cloud(N, UnA_N, VY, Tt, Adsorbed, I, I, -3.0, 3.0, 0.0, 50.0)
    #Cloud_dat('VY', 'Tt', I, I, vi, tv, Rfvit)
    wi, tw, Rfwit = Distribution_Cloud(N, UnA_N, VZ, Tt, Adsorbed, I, I, -3.0, 0.0, 0.0, 50.0)
    #Cloud_dat('VZ', 'Tt', I, I, wi, tw, Rfwit)
    #
    ui, vr, Rfuivr = Distribution_Cloud(N, UnA_N, VX, FVY, Adsorbed, I, I, -3.0, 3.0, -3.0, 3.0)
    #Cloud_dat('VX', 'FVY', I, I, ui, vr, Rfuivr)
    ui, wr, Rfuiwr = Distribution_Cloud(N, UnA_N, VX, FVZ, Adsorbed, I, I, -3.0, 3.0, 0.0, 3.0)
    #Cloud_dat('VX', 'FVZ', I, I, ui, wr, Rfuiwr)
    #
    ui, ur, Rfuir = Distribution_Cloud(N, UnA_N, VX, FVX, Adsorbed, I, I, -3.0, 3.0, -3.0, 3.0)
    #Cloud_dat('VX', 'FVX', I, I, ui, ur, Rfuir)
    vi, vr, Rfvir = Distribution_Cloud(N, UnA_N, VY, FVY, Adsorbed, I, I, -3.0, 3.0, -3.0, 3.0)
    #Cloud_dat('VY', 'FVY', I, I, vi, vr, Rfvir)
    wi, wr, Rfwir = Distribution_Cloud(N, UnA_N, VZ, FVZ, Adsorbed, I, I, -3.0, 0.0, 0.0, 3.0)
    #Cloud_dat('VZ', 'FVZ', I, I, wi, wr, Rfwir)
    Swi, Swr, SRfwir = Distribution_Cloud(SamplesN, SamplesUAN, SampleVZ, SampleFVZ, SampleAds, I, I, -3.0, 0.0, 0.0, 3.0)
    #Cloud_dat('VZ', 'FVZ', I, I, wi, wr, Rfwir, 'Sample', 0, SRfwir)
    wi, wr, EwnExcMat, EwtExcMat, EwTExcMat, ERwntInMat, ERwntOutMat, ERwntExcMat = Energy_Distribution_Cloud(N, VZ, FVZ, VX, FVX, VY, FVY, Adsorbed, I, I, -3.0, 0.0, 0.0, 3.0)
    Cloud_dat('VZ', 'FVZ', I, I, wi, wr, EwnExcMat, 'EEn', 'Energy', Rfwir)
    Cloud_dat('VZ', 'FVZ', I, I, wi, wr, EwtExcMat, 'EEt', 'Energy', Rfwir)
    Cloud_dat('VZ', 'FVZ', I, I, wi, wr, EwTExcMat, 'EET', 'Energy', Rfwir)
    Cloud_dat('VZ', 'FVZ', I, I, wi, wr, ERwntInMat, 'ERntIn', 'Energy', Rfwir)
    Cloud_dat('VZ', 'FVZ', I, I, wi, wr, ERwntOutMat, 'ERntOut', 'Energy', Rfwir)
    Cloud_dat('VZ', 'FVZ', I, I, wi, wr, ERwntExcMat, 'ERntExc', 'Energy', Rfwir)
    #
    u, fui, fur, fua, fuUa = Distribution(N, UnA_N, VX, FVX, Adsorbed, I, -3.0, 3.0)
    v, fvi, fvr, fva, fvUa = Distribution(N, UnA_N, VY, FVY, Adsorbed, I, -3.0, 3.0)
    #Distribution_Plot('VY', v, fvi, fvr, fva, -3.0, 3.0, 0.0, 1.0)
    w, fwi, fwr, fwa, fwUa = Distribution(N, UnA_N, VZ, FVZ, Adsorbed, 2 * I, -3.0, 3.0)
    #Distribution_Plot('VZ', w, fwi, fwr, fwa, -3.0, 3.0, 0.0, 1.4)
    #
    Speed = GetSpeed(N, VX, VY, VZ)
    AP_u, u_AP = AP_Distribution(N, VX, Adsorbed, I, -3.0, 3.0)
    AP_v, v_AP = AP_Distribution(N, VY, Adsorbed, I, -3.0, 3.0)
    AP_w, w_AP = AP_Distribution(N, VZ, Adsorbed, 2 * I, -3.0, 3.0)
    AP_Sp, Sp_AP = AP_Distribution(N, Speed, Adsorbed, 2 * I, 0.0, 5.0)
    AP_dat(AP_u, u_AP, v_AP, w_AP, AP_Sp, Sp_AP, 'local')
    AP_Distribution_Plot(AP_u, AP_v, AP_w, AP_Sp, u_AP, v_AP, w_AP, Sp_AP, 'local')
    AP_Sp, Sp_AP = AP_Distribution(N, Speed, Adsorbed, 8 * I, 0.0, 5.0)
    AP_dat(AP_u, u_AP, v_AP, w_AP, AP_Sp, Sp_AP, 'global')
    AP_Distribution_Plot(AP_u, AP_v, AP_w, AP_Sp, u_AP, v_AP, w_AP, Sp_AP, 'global')
    #
    AP_Sp, AP_u, Su_AP = AP_Distribution_Cloud(N, Speed, VX, Adsorbed, 2 * I, I, 0.0, 5.0, -3.0, 3.0)
    Cloud_dat('Speed', 'VX', 2 * I, I, AP_Sp, AP_u, Su_AP)
    AP_Sp, AP_v, Sv_AP = AP_Distribution_Cloud(N, Speed, VY, Adsorbed, 2 * I, I, 0.0, 5.0, -3.0, 3.0)
    Cloud_dat('Speed', 'VY', 2 * I, I, AP_Sp, AP_v, Sv_AP)
    AP_Sp, AP_w, Sw_AP = AP_Distribution_Cloud(N, Speed, VZ, Adsorbed, 2 * I, 2 * I, 0.0, 5.0, -3.0, 3.0)
    Cloud_dat('Speed', 'VZ', 2 * I, 2 * I, AP_Sp, AP_w, Sw_AP)
    AP_u, AP_w, uw_AP = AP_Distribution_Cloud(N, VX, VZ, Adsorbed, I, 2 * I, -3.0, 3.0, -3.0, 3.0)
    Cloud_dat('VX', 'VZ', I, 2 * I, AP_u, AP_w, uw_AP)
    #
    for i in range(I + 1):
        if fuUa[i] != 0.0:
            for j in range(I + 1):
                Rfuit[i, j] /= fuUa[i]
    Cloud_dat('VX', 'Tt', I, I, ui, tu, Rfuit)
    for i in range(I + 1):
        if fvUa[i] != 0.0:
            for j in range(I + 1):
                Rfvit[i, j] /= fvUa[i]
    Cloud_dat('VY', 'Tt', I, I, vi, tv, Rfvit)
    for i in range(I + 1):
        if fwUa[i] != 0.0:
            for j in range(I + 1):
                Rfwit[i, j] /= fwUa[i]
    Cloud_dat('VZ', 'Tt', I, I, wi, tw, Rfwit)
    #
    #FIG8_Distribution_Plot(GasTGasmpV, w, fur, fvr, fwr)
    User_lt = np.array([0.909, 2.639, 0.095, 1.0, 0.0, 0.0])
    User_Rfuivr = Irrelevant(CLL_R, 0, N, ui, vr, vi, wi, fui, fvi, fwi, GasTGasmpV, WallTGasmpV, User_lt)
    Cloud_dat('VX', 'CLL_FVY', I, I, ui, vr, User_Rfuivr, CloudName1 = 'f', Type = 'CLL', CloudDat0 = Rfuivr, CloudName0 = 'MD', Loop = 0)
    #User_lt = np.array([0.054, 0.546, 0.173, 0.827, 0.0, 0.0])
    #User_Rfuiwr = Irrelevant(CLL_R, 1, N, ui, wr, vi, wi, fui, fvi, fwi, GasTGasmpV, WallTGasmpV, User_lt)
    #Cloud_dat('VX', 'CLL_FVZ', I, I, ui, wr, User_Rfuiwr, CloudName1 = 'f', Type = 'CLL', CloudDat0 = Rfuiwr, CloudName0 = 'MD', Loop = 0)
    #
    Loops = 25000
    mu = 0.999
    #
    lt_N = 6
    lt_x = np.zeros(lt_N)
    v_x = np.zeros(lt_N)
    lt_y = np.zeros(lt_N)
    v_y = np.zeros(lt_N)
    lt_z = np.zeros(lt_N)
    v_z = np.zeros(lt_N)
    lt_x[:3] = 0.5
    lt_y[:3] = 1.5
    lt_z[:3] = 0.5
    #ltwi_z = np.ones(len(ui))
    #vwi_z = np.zeros(len(ui))
    #lt_x[3] = 1.0
    #lt_y[3] = 1.0
    #lt_z[2] = 1.0
    #
    '''
    lt1_x = 0.5
    lt2_x = 0.0
    lt3_x = 0.0
    lt4_x = 0.0
    lt5_x = 0.0
    lt6_x = 0.0
    v1_x = 0.0
    v2_x = 0.0
    v3_x = 0.0
    v4_x = 0.0
    v5_x = 0.0
    v6_x = 0.0
    lt1_y = 1.5
    lt2_y = 0.0
    lt3_y = 0.0
    lt4_y = 0.0
    lt5_y = 0.0
    lt6_y = 0.0
    v1_y = 0.0
    v2_y = 0.0
    v3_y = 0.0
    v4_y = 0.0
    v5_y = 0.0
    v6_y = 0.0
    lt1_z = 0.3
    lt2_z = 0.7
    lt3_z = 1.0
    lt4_z = 0.333
    lt5_z = 0.333
    lt6_z = 0.333
    v1_z = 0.0
    v2_z = 0.0
    v3_z = 0.0
    v4_z = 0.0
    v5_z = 0.0
    v6_z = 0.0
    '''
    #
    Loop = np.zeros(Loops + 1)
    Err_X = np.zeros(Loops + 1)
    Err_Y = np.zeros(Loops + 1)
    Err_Z = np.zeros(Loops + 1)
    lt_X = np.zeros((Loops + 1, lt_N))
    lt_Y = np.zeros((Loops + 1, lt_N))
    lt_Z = np.zeros((Loops + 1, lt_N))
    #
    '''
    lt1_X=np.zeros(Loops+1)
    lt2_X=np.zeros(Loops+1)
    lt3_X=np.zeros(Loops+1)
    lt4_X=np.zeros(Loops+1)
    lt5_X=np.zeros(Loops+1)
    lt6_X=np.zeros(Loops+1)
    lt1_Y=np.zeros(Loops+1)
    lt2_Y=np.zeros(Loops+1)
    lt3_Y=np.zeros(Loops+1)
    lt4_Y=np.zeros(Loops+1)
    lt5_Y=np.zeros(Loops+1)
    lt6_Y=np.zeros(Loops+1)
    lt1_Z=np.zeros(Loops+1)
    lt2_Z=np.zeros(Loops+1)
    lt3_Z=np.zeros(Loops+1)
    lt4_Z=np.zeros(Loops+1)
    lt5_Z=np.zeros(Loops+1)
    lt6_Z=np.zeros(Loops+1)
    '''
    #
    ACXMat = np.ones((len(ui), len(ur))) * 0.5
    VXMat = np.zeros((len(ui), len(ur)))
    ACYMat = np.ones((len(vi), len(vr))) * 1.5
    VYMat = np.zeros((len(vi), len(vr)))
    ACZMat = np.ones((len(wi), len(wr))) * 0.5
    VZMat = np.zeros((len(wi), len(wr)))
    #
    #ACX_Vec = np.ones(len(ui)) * 0.5
    #VX_Vec = np.zeros(len(ui))
    #ACY_Vec = np.ones(len(vi)) * 1.5
    #VY_Vec = np.zeros(len(vi))
    #ACZ_Vec = np.ones(len(wi)) * 0.5
    #VZ_Vec = np.zeros(len(wi))
    #
    #PAX_Vec = np.ones((len(ui), 3))
    #VX_Vec = np.zeros((len(ui), 3))
    #PAY_Vec = np.ones((len(vi), 3))
    #VY_Vec = np.zeros((len(vi), 3))
    #PAZ_Vec = np.ones((len(wi), 3)) * 0.5
    #VZ_Vec = np.zeros((len(wi), 3))
    #
    with open('Fitting-X.log', 'w') as out_x:
        tile = 'Loop\tErrX'
        for i in range(len(lt_x)):
            tile += '\tlt' + str(i) + '_X'
        print(tile, file = out_x)
    #with open('Fitting-Y.log', 'w') as out_y:
    #    tile = 'Loop\tErrY'
    #    for i in range(len(lt_y)):
    #        tile += '\tlt' + str(i) + '_Y'
    #    print(tile, file = out_y)
    #with open('Fitting-Z.log', 'w') as out_z:
    #    tile = 'Loop\tErrZ'
    #    for i in range(len(lt_z)):
    #        tile += '\tlt' + str(i) + '_Z'
    #    print(tile, file = out_z)
    #with open('Fitting-Z-wi.log', 'w') as out_z:
    #    print('Loop\tltwi', file = out_z)
    #
    ui, vi, wi, ur, Rfiur = Distribution_Cloud_FourPhase(N, UnA_N, VX, VY, VZ, FVX, Adsorbed, I, I, -3.0, 3.0, -3.0, 0.0, -3.0, 3.0)
    #ui, vi, wi, vr, Rfivr = Distribution_Cloud_FourPhase(N, UnA_N, VX, VY, VZ, FVY, Adsorbed, I, I, -3.0, 3.0, -3.0, 0.0, -3.0, 3.0)
    #ui, vi, wi, wr, Rfiwr = Distribution_Cloud_FourPhase(N, UnA_N, VX, VY, VZ, FVZ, Adsorbed, I, I, -3.0, 3.0, -3.0, 0.0, 0.0, 3.0)
    #
    for l in range(Loops + 1):
        CLL_Rfuiur, CLL_Noru, CLL_Fu, CLL_fui, CLL_fur, Errx, lt_x, v_x = Gradient_Least_Squares_FourPhase(CLL_R, 0, 'L1', ui, vi, wi, ur, Rfiur, fuUa, fvUa, fwUa, GasTGasmpV, WallTGasmpV, mu, lt_x, v_x)
        #CLL_Rfu, CLL_Noru, CLL_Fu, CLL_fui, CLL_fur, Errx, lt_x, v_x = Gradient_Least_Squares(CLL_R, 0, ui, vi, wi, ur, Rfuir, fuUa, fvUa, fwUa, GasTGasmpV, WallTGasmpV, mu, lt_x, v_x)
        #CLL_Rfv, CLL_Norv, CLL_Fv, CLL_fvi, CLL_fvr, Erry, lt_y, v_y = Gradient_Least_Squares(CLL_R, 0, vi, ui, wi, vr, Rfvir, fvUa, fuUa, fwUa, GasTGasmpV, WallTGasmpV, mu, lt_y, v_y)
        #if l % 2 == 0:
        #    CLL_Rfw, CLL_Norw, CLL_Fw, CLL_fwi, CLL_fwr, Errz, lt_z, v_z, ltwi_z, vwi_z = Gradient_Least_Squares(CLL_R, 1, wi, ui, vi, wr, Rfwir, fwUa, fuUa, fvUa, GasTGasmpV, WallTGasmpV, mu, lt_z, v_z, ltwi_z, vwi_z)
        #Eps_Rfw,Eps_Norw,Eps_fw,Eps_fwi,Eps_fwr,Errz,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z=Gradient_Least_Squares(Epstein,1,N,wi,ui,vi,wr,Rfwir,fwUa,fuUa,fvUa,VZ,FVZ,VX,VY,Adsorbed,WallTGasmpV,WallTGasmpV,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,mu,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z)
        #Rec_Rfw,Rec_Norw,Rec_Fw,Rec_fwi,Rec_fwr,Errz,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z=Gradient_Least_Squares(Reciprocal,1,N,wi,ui,vi,wr,Rfwir,fwUa,fuUa,fvUa,VZ,FVZ,VX,VY,Adsorbed,WallTGasmpV,WallTGasmpV,lt1_z,lt2_z,lt3_z,lt4_z,lt5_z,lt6_z,mu,v1_z,v2_z,v3_z,v4_z,v5_z,v6_z)
        #CLL_Rfu, ErrACX, PAX_Vec, VX_Vec = AC_Vec(0, ui, ur, Rfuir, fuUa, GasTGasmpV, WallTGasmpV, PAX_Vec, VX_Vec)
        #CLL_Rfv, ErrACY, PAY_Vec, VY_Vec = AC_Vec(0, vi, vr, Rfvir, fvUa, GasTGasmpV, WallTGasmpV, PAY_Vec, VY_Vec)
        #CLL_Rfw, ErrACZ, PAZ_Vec, VZ_Vec = AC_Vec(1, wi, wr, Rfwir, fwUa, GasTGasmpV, WallTGasmpV, PAZ_Vec, VZ_Vec)
        #CLL_fui=np.zeros(len(fui))
        #CLL_fur=np.zeros(len(fur))
        #CLL_Noru=np.zeros(len(fui))
        #CLL_fvi=np.zeros(len(fvi))
        #CLL_fvr=np.zeros(len(fvr))
        #CLL_Norv=np.zeros(len(fvi))
        #CLL_fwi=np.zeros(len(fwi))
        #CLL_fwr=np.zeros(len(fwr))
        #CLL_Norw=np.zeros(len(fwi))
        print(l)
        #print(Errx, lt_x, CLL_Fu)
        #print(Erry, lt_y, CLL_Fv)
        #print(Errz, lt_z, CLL_Fw)
        #print(l, ErrACX, ErrACY, ErrACZ)
        with open('Fitting-X.log', 'a') as out_x:
            print('%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f' % (l, Errx, lt_x[0], lt_x[1], lt_x[2], lt_x[3], lt_x[4], lt_x[5]), file = out_x)
        #with open('Fitting-Y.log', 'a') as out_y:
        #    print('%d\t%f\t%f\t%f\t%f\t%f' % (l, Erry, lt_y[0], lt_y[1], lt_y[2], lt_y[3]), file = out_y)
        #with open('Fitting-Z.log', 'a') as out_z:
        #    print('%d\t%f\t%f\t%f' % (l, Errz, lt_z[0], lt_z[1]), file = out_z)
        #with open('Fitting-Z-wi.log', 'a') as out_z:
        #    print(l, ltwi_z, file = out_z)
        Loop[l] = l
        Err_X[l] = Errx
        for p in range(len(lt_x)):
            lt_X[l, p] = lt_x[p]
        #Err_Y[l] = Erry
        #for p in range(len(lt_y)):
        #    lt_Y[l, p] = lt_y[p]
        #Err_Z[l] = Errz
        #for p in range(len(lt_z)):
        #    lt_Z[l, p] = lt_z[p]
        if l % 100 == 0:
            Cloud_dat('VX', 'CLL_FVX', I, I, ui, ur, CLL_Rfuiur, 'Rfu', 'CLL', Rfuir, 'MD', l)
            #Cloud_dat('VY', 'CLL_FVY', I, I, vi, vr, CLL_Rfv, 'Rfv', 'CLL', Rfvir, 'MD', l)
            #if l % 200 == 0:
            #    Cloud_dat('VZ', 'CLL_FVZ', I, I, wi, wr, CLL_Rfw, 'Rfw', 'CLL', Rfwir, 'MD', l)
            #Cloud_dat('VX', 'FVX', I, I, ui, ur, ACXMat, 'ACu', 'AC', Rfuir, 'MD', l)
            #Cloud_dat('VY', 'FVY', I, I, vi, vr, ACYMat, 'ACv', 'AC', Rfvir, 'MD', l)
            #Cloud_dat('VZ', 'FVZ', I, I, wi, wr, ACZMat, 'ACw', 'AC', Rfwir, 'MD', l)
    #
    '''
    with open('Fitting_PAX_Vec.dat', 'w') as out:
        for i in range(len(PAX_Vec)):
            print(ui[i, 1], PAX_Vec[i, 0], PAX_Vec[i, 1], PAX_Vec[i, 2], file = out)
    with open('Fitting_PAY_Vec.dat', 'w') as out:
        for i in range(len(PAY_Vec)):
            print(vi[i, 1], PAY_Vec[i, 0], PAY_Vec[i, 1], PAY_Vec[i, 2], file = out)
    with open('Fitting_PAZ_Vec.dat', 'w') as out:
        for i in range(len(PAZ_Vec)):
            print(wi[i, 1], PAZ_Vec[i, 0], PAZ_Vec[i, 1], PAZ_Vec[i, 2], file = out)
    '''
    Err_Plot('X', Loop, Err_X, lt_X)
    #Err_Plot('Y', Loop, Err_Y, lt_Y)
    #Err_Plot('Z', Loop, Err_Z, lt_Z)
    #FIG12_Distribution_Plot(w,fwUa,fwr,CLL_fwi,CLL_fwr,CLL_Norw,u,fuUa,fur,CLL_fui,CLL_fur,CLL_Noru,fvUa,fvr,CLL_fvi,CLL_fvr,CLL_Norv)
    #Distribution_Plot('VX',u,fui,fur,fua,fuUa,CLL_fui,CLL_fur,CLL_Noru,-3.0,3.0,0.0,0.7,GasT)
    #Distribution_Plot('VY',v,fvi,fvr,fva,fvUa,CLL_fvi,CLL_fvr,CLL_Norv,-3.0,3.0,0.0,0.7,GasT)
    #Distribution_Plot('VZ',w,fwi,fwr,fwa,fwUa,CLL_fwi,CLL_fwr,CLL_Norw,-3.0,3.0,0.0,1.0,GasT)
    #
    for i in range(I + 1):
        if fuUa[i] != 0.0:
            for j in range(I + 1):
                Rfuivr[i, j] /= fuUa[i]
    Cloud_dat('VX', 'FVY', I, I, ui, vr, Rfuivr)
    for i in range(I + 1):
        if fuUa[i] != 0.0:
            for j in range(I + 1):
                Rfuiwr[i, j] /= fuUa[i]
    Cloud_dat('VX', 'FVZ', I, I, ui, wr, Rfuiwr)
    #
    for i in range(I + 1):
        if fuUa[i] != 0.0:
            for j in range(I + 1):
                Rfuir[i, j] /= fuUa[i]
    Cloud_dat('VX', 'FVX', I, I, ui, ur, Rfuir)
    for i in range(I + 1):
        if fvUa[i] != 0.0:
            for j in range(I + 1):
                Rfvir[i, j] /= fvUa[i]
    Cloud_dat('VY', 'FVY', I, I, vi, vr, Rfvir)
    for i in range(I + 1):
        if fwUa[i] != 0.0:
            for j in range(I + 1):
                Rfwir[i, j] /= fwUa[i]
    Cloud_dat('VZ', 'FVZ', I, I, wi, wr, Rfwir)
    '''
    #K_Means
    #X, Y, Z, VX, VY, VZ, Adsorbed
    K = 2
    KM_N = len(X)
    K_Means_Data = X.copy().reshape((KM_N, 1))
    K_Means_Data = np.append(K_Means_Data, Y.copy().reshape((KM_N, 1)), axis = 1)
    K_Means_Data = np.append(K_Means_Data, Z.copy().reshape((KM_N, 1)), axis = 1)
    K_Means_Data = np.append(K_Means_Data, VX.copy().reshape((KM_N, 1)), axis = 1)
    K_Means_Data = np.append(K_Means_Data, VY.copy().reshape((KM_N, 1)), axis = 1)
    K_Means_Data = np.append(K_Means_Data, VZ.copy().reshape((KM_N, 1)), axis = 1)
    KM_BelongInf, KM_Clusters = ML_Analysis.Bi_K_Means_Wong(K_Means_Data, K)
    fig = mpl.figure()
    ax = fig.add_subplot(111, projection = '3d')
    KM_N, KM_Dim = K_Means_Data.shape
    markers = ['o', 'v', '^', 's', 'P', '*', 'D', 'X']
    colours = ['r', 'b', 'g', 'c', 'm', 'y', 'tomato', 'skyblue']
    for i in range(KM_N):
        ax.scatter(K_Means_Data[i, 3], K_Means_Data[i, 4], K_Means_Data[i, 5], s = 2, c = colours[int(KM_BelongInf[i, 0])], marker = markers[int(KM_BelongInf[i, 0])])
    ax.scatter(KM_Clusters[:, 4], KM_Clusters[:, 5], KM_Clusters[:, 6], s = 20, c = 'k', marker = 'x')
    #ax.set_xlim(-9.0, 9.0)
    #ax.set_ylim(-9.0, 9.0)
    #ax.set_zlim(-9.0, 9.0)
    mpl.savefig('Bi_K_Means_Wong.png', dpi = 600)
    #fig.show()
    mpl.close()
    '''

################################################################################
if __name__ == '__main__':
    #
    main()
