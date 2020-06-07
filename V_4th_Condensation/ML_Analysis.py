import numpy as np
import numba as nb
import matplotlib.pyplot as mpl
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

################################################################################
def ReadFile(FileName):
    #
    with open(FileName, 'r') as In:
        Data = In.readlines()
        InData = Data[1:]
        AP_U = []
        AP_V = []
        AP_W = []
        U_AP = []
        V_AP = []
        W_AP = []
        for pdata in InData:
            (uvw, u_ap, v_ap, w_ap) = pdata.split('\t', 3)
            if not np.isnan(float(u_ap)):
                AP_U.append(float(uvw))
                U_AP.append(float(u_ap))
            if not np.isnan(float(v_ap)):
                AP_V.append(float(uvw))
                V_AP.append(float(v_ap))
            if not np.isnan(float(w_ap)):
                AP_W.append(float(uvw))
                W_AP.append(float(w_ap))
    AP_U = np.array(AP_U)
    AP_V = np.array(AP_V)
    AP_W = np.array(AP_W)
    U_AP = np.array(U_AP)
    V_AP = np.array(V_AP)
    W_AP = np.array(W_AP)
    #
    return(AP_U, AP_V, AP_W, U_AP, V_AP, W_AP)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Fitting2D(X, Y, Model, params, paramsLims, LOOPS, Type):
    #
    print(Type)
    NX = len(X)
    NY = len(Y)
    Np = len(params)
    dp = 1e-8
    mu = 0.999
    alpha = 1e-4
    Loops = LOOPS
    params_Loop = np.zeros((Loops + 1, Np))
    Error_Loop = np.zeros(Loops + 1)
    #
    if NX == NY:
        Predict_Y = np.zeros((NY, Np + 1))
        dErrdp = np.zeros(Np)
        v = np.zeros(Np)
        #
        for l in range(Loops + 1):
            Error = np.zeros(Np + 1)
            for i in range(NX):
                for j in range(Np + 1):
                    params_copy = params.copy()
                    if j != 0:
                        params_copy[j - 1] += dp
                    Predict_Y[i, j] = Model(X[i], params_copy)
                    Error[j] += (Predict_Y[i, j] - Y[i]) ** 2
            #
            Error_Loop[l] = Error[0]
            for j in range(Np):
                params_Loop[l, j] = params[j]
                dErrdp[j] = (Error[j + 1] - Error[0]) / dp
                #
                pre_v = v[j]
                v[j] = mu * v[j]
                v[j] += - alpha * dErrdp[j]
                params[j] += v[j] + mu * (v[j] - pre_v)
                #
                if params[j] <= paramsLims[j, 0]:
                    params[j] = paramsLims[j, 0] + dp
                    v[j] = - v[j]
                elif params[j] >= paramsLims[j, 1] - dp:
                    params[j] = paramsLims[j, 1] - 2 * dp
                    v[j] = - v[j]
    #
    return(params, params_Loop, Error_Loop)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Maxwellian_T(X, params):
    #params: 0.标准差 = sqrt(kT/m) (正值); 1.调节系数(理论上应为1?)
    Y = params[1] / (params[0] * np.sqrt(2 * np.pi)) * np.exp(- X ** 2 / (2 * params[0] ** 2))
    #
    return(Y)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Maxwellian_N(X, params):
    #params: 0.标准差 = sqrt(kT/m) (正值); 1.调节系数 (理论上应为1?); 2.偏移(使速度为零时不为零)
    Y = (params[1] / params[0]) ** 2 * (abs(X) ** 0.5 + params[2]) * np.exp(- X ** 2 / (2 * params[0] ** 2))
    #
    return(Y)

################################################################################
#@nb.jit(nopython = True, nogil = True)
def Fitting2D_Plot(X1, X2, X3, Y1, Y2, Y3, P1, P2, P3, Model1, Model2, Model3):
    #
    Predict_N1 = 2 * len(X1)
    Predict_X1 = np.zeros(Predict_N1 + 1)
    Predict_Y1 = np.zeros(Predict_N1 + 1)
    X1_Low = np.min(X1)
    X1_High = np.max(X1)
    dX1 = (X1_High - X1_Low) / Predict_N1
    for i in range(Predict_N1 + 1):
        Predict_X1[i] = X1_Low + i * dX1
        Predict_Y1[i] = Model1(Predict_X1[i], P1)
    #
    Predict_N2 = 2 * len(X2)
    Predict_X2 = np.zeros(Predict_N2 + 1)
    Predict_Y2 = np.zeros(Predict_N2 + 1)
    X2_Low = np.min(X2)
    X2_High = np.max(X2)
    dX2 = (X2_High - X2_Low) / Predict_N2
    for i in range(Predict_N2 + 1):
        Predict_X2[i] = X2_Low + i * dX2
        Predict_Y2[i] = Model2(Predict_X2[i], P2)
    #
    Predict_N3 = 2 * len(X3)
    Predict_X3 = np.zeros(Predict_N3 + 1)
    Predict_Y3 = np.zeros(Predict_N3 + 1)
    X3_Low = np.min(X3)
    X3_High = np.max(X3)
    dX3 = (X3_High - X3_Low) / Predict_N3
    for i in range(Predict_N3 + 1):
        Predict_X3[i] = X3_Low + i * dX3
        Predict_Y3[i] = Model3(Predict_X3[i], P3)
    #
    lab1 = 'X-Adsorption Probability Distribution'
    lab2 = 'Y-Adsorption Probability Distribution'
    lab3 = 'Z-Adsorption Probability Distribution'
    #
    fig0, ax0 = mpl.subplots()
    yaxis_high = max(np.max(Y1), max(np.max(Y2), np.max(Y3)))
    yaxis_high *= 1.2
    ax0.plot(X1, Y1, 'ro', markersize = 3, label = lab1)
    ax0.plot(X2, Y2, 'bv', markersize = 3, label = lab2)
    ax0.plot(X3, Y3, 'mD', markersize = 3, label = lab3)
    ax0.plot(Predict_X1, Predict_Y1, 'r-', markersize = 3)
    ax0.plot(Predict_X2, Predict_Y2, 'b-', markersize = 3)
    ax0.plot(Predict_X3, Predict_Y3, 'm-', markersize = 3)
    #
    ax0.legend(loc = 'upper right', fontsize = 'small', frameon = False)
    ax0.set_xlabel('$u, v, w$')
    ax0.set_ylabel('$Adsorption Probability$')
    ax0.set_xlim(-3.0, 3.0)
    ax0.set_ylim(0.0, yaxis_high)
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
    ax1.set_ylim(0.0, yaxis_high)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.tick_params(axis = 'y', which = 'major', direction = 'in', length = 5, width = 2)
    ax1.tick_params(axis = 'y', which = 'minor', direction = 'in', length = 3, width = 1)
    ax1.spines['right'].set_linewidth(2)
    #
    ax2 = ax0.twiny()
    ax2.set_xlim(-3.0, 3.0)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.tick_params(axis = 'x', which = 'major', direction = 'in', length = 5, width = 2)
    ax2.tick_params(axis = 'x', which = 'minor', direction = 'in', length = 3, width = 1)
    ax2.spines['top'].set_linewidth(2)
    #
    mpl.savefig('AP_Fitting_uvw.png', dpi = 600)

################################################################################
def Fitting2D_Loops_dat(PL, EL, Type):
    #
    Loops, NP = PL.shape
    with open('AP_Fitting_Loops_' + Type + '.dat','w') as Out:
        if Type == 'w':
            print('%s\t%s\t%s\t%s\t%s' % ('Loop', 'Param0', 'Param1', 'Param2', 'Error'), file = Out)
        else:
            print('%s\t%s\t%s\t%s' % ('Loop', 'Param0', 'Param1', 'Error'), file = Out)
        for i in range(Loops):
            if Type == 'w':
                print('%d\t%f\t%f\t%f\t%f' % (i, PL[i, 0], PL[i, 1], PL[i, 2], EL[i]), file = Out)
            else:
                print('%d\t%f\t%f\t%f' % (i, PL[i, 0], PL[i, 1], EL[i]), file = Out)

################################################################################
def Fitting2D_Loops_Plot(PL, EL, Type):
    #
    markers = ['r', 'b', 'm', 'c']
    Loops, NP = PL.shape
    l = [i for i in range(Loops)]
    l = np.array(l)
    labs = []
    for i in range(NP):
        labs.append('param_' + str(i))
    labs.append('Error')
    #
    fig0, ax0 = mpl.subplots()
    for i in range(NP):
        ax0.plot(l, PL[:, i], markers[i] + '-', markersize = 3, label = labs[i])
    ax0.plot(l, EL, markers[i + 1] + '-', markersize = 3, label = labs[i + 1])
    #
    ax0.legend(loc = 'upper right', fontsize = 'small', frameon = False)
    ax0.set_xlabel('$Loop$')
    ax0.set_ylabel('$Error, Params$')
    ax0.set_xlim(0.0, Loops - 1.0)
    #
    mpl.savefig('AP_Fitting_Loops_' + Type + '.png', dpi = 600)

################################################################################
@nb.jit(nopython = True, nogil = True)
def K_Means_Distance(ClusterPos, PointPos):
    #
    Dim = len(ClusterPos)
    if len(PointPos) == Dim:
        Dis = 0.0
        for i in range(Dim):
            Dis += (ClusterPos[i] - PointPos[i]) ** 2
        #Dis = np.sqrt(Dis)
    #
    return(Dis)

################################################################################
@nb.jit(nopython = True, nogil = True)
def K_Means_InitialCluster(Data, K):
    #
    N, Dim = Data.shape
    #
    Clusters = np.zeros((K, Dim + 1))                                               #簇大小; 簇位置
    for j in range(Dim):
        LowLim = np.min(Data[:, j])
        HighLim = np.max(Data[:, j])
        Clusters[:, j + 1] = LowLim + (HighLim - LowLim) * np.random.random(K)
    #
    return(Clusters)

################################################################################
@nb.jit(nopython = True, nogil = True)
def InputClusters(Clusters, K):
    #
    return(Clusters)

################################################################################
@nb.jit(nopython = True, nogil = True)
def K_Means(Data, K, InputBelongInf = None, InputClusters = None, DistMethod = K_Means_Distance, InitMethod = K_Means_InitialCluster):
    #
    N, Dim = Data.shape
    #
    if InputBelongInf is None:
        BelongInf = np.zeros((N, 2))
    else:
        BelongInf = InputBelongInf                                              #所属簇编号; 距离平方
    if InputClusters is None:
        Clusters = InitMethod(Data, K)                                          #簇大小; 簇位置
    else:
        Clusters = InputClusters
    ClusterChanged = True
    #
    while ClusterChanged:
        ClusterChanged = False
        TempClusters = np.zeros((K, Dim + 1))                                   #簇大小; 簇位置
        for i in range(N):
            NearestDis = np.inf
            NearestCluster = -1
            for k in range(K):
                Dis = DistMethod(Clusters[k, 1:], Data[i, :])
                if Dis < NearestDis:
                    NearestDis = Dis
                    NearestCluster = k
            if BelongInf[i, 0] != NearestCluster:
                ClusterChanged = True
            BelongInf[i, 0] = NearestCluster
            BelongInf[i, 1] = NearestDis
            TempClusters[int(BelongInf[i, 0]), 0] += 1
            TempClusters[int(BelongInf[i, 0]), 1:] += Data[i, :]
        #
        for k in range(K):
            if TempClusters[k, 0] == 0:
                TempClusters[k, 1:] = Clusters[k, 1:]
            else:
                TempClusters[k, 1:] /= TempClusters[k, 0]
        Clusters[:, :] = TempClusters[:, :]
    #
    return(BelongInf, Clusters)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Bi_K_Means(Data, K, DistMethod = K_Means_Distance):
    #
    N, Dim = Data.shape
    BelongInf = np.zeros((N, 2))
    Clusters = np.zeros((K, Dim + 1))
    #
    Clusters[0, 0] = N
    for j in range(Dim):
        for i in range(N):
            Clusters[0, j + 1] += Data[i, j]
        Clusters[0, j + 1] /= Clusters[0, 0]
    #
    for i in range(N):
        BelongInf[i, 0] = 0
        BelongInf[i, 1] = DistMethod(Clusters[0, 1:], Data[i, :])
    #
    k = 1
    while k < K:
        #
        print(Clusters)
        LowestSSE = np.inf
        for i in range(k):
            #
            ClusterData_tosplit = np.zeros((int(Clusters[i, 0]), Dim))
            count = 0
            SSE_notsplit = 0.0
            for n in range(N):
                if BelongInf[n, 0] == i:
                    ClusterData_tosplit[count, :] = Data[n, :]
                    count += 1
                else:
                    SSE_notsplit += BelongInf[n, 1]
            #
            BelongInf_tosplit, Clusters_tosplit = K_Means(ClusterData_tosplit, 2)
            SSE_tosplit = np.sum(BelongInf_tosplit[:, 1])
            #
            if SSE_tosplit + SSE_notsplit < LowestSSE:
                BestToSplit_ClusterID = i
                BestToSplit_BelongInf = BelongInf_tosplit.copy()
                BestToSplit_Clusters = Clusters_tosplit.copy()
                LowestSSE = SSE_tosplit + SSE_notsplit
        #
        count = 0
        for j in range(N):
            if BelongInf[j, 0] == BestToSplit_ClusterID:
                if BestToSplit_BelongInf[count, 0] == 0:
                    BelongInf[j, 0] = BestToSplit_ClusterID
                    BelongInf[j, 1] = BestToSplit_BelongInf[count, 1]
                elif BestToSplit_BelongInf[count, 0] == 1:
                    BelongInf[j, 0] = k
                    BelongInf[j, 1] = BestToSplit_BelongInf[count, 1]
                count += 1
        #
        Clusters[BestToSplit_ClusterID, :] = BestToSplit_Clusters[0, :]
        Clusters[k, :] = BestToSplit_Clusters[1, :]
        #
        k += 1
    #
    print(Clusters)
    return(BelongInf, Clusters)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Bi_K_Means_Wong(Data, K, DistMethod = K_Means_Distance):
    #
    N, Dim = Data.shape
    BelongInf = np.zeros((N, 2))
    Clusters = np.zeros((K, Dim + 1))
    #
    Clusters[0, 0] = N
    for j in range(Dim):
        for i in range(N):
            Clusters[0, j + 1] += Data[i, j]
        Clusters[0, j + 1] /= Clusters[0, 0]
    #
    for i in range(N):
        BelongInf[i, 0] = 0
        BelongInf[i, 1] = DistMethod(Clusters[0, 1:], Data[i, :])
    #
    k = 1
    while k < K:
        #
        print(Clusters)
        LowestSSE = np.inf
        for i in range(k):
            #
            ClusterData_tosplit = np.zeros((int(Clusters[i, 0]), Dim))
            count = 0
            SSE_notsplit = 0.0
            for n in range(N):
                if BelongInf[n, 0] == i:
                    ClusterData_tosplit[count, :] = Data[n, :]
                    count += 1
                else:
                    SSE_notsplit += BelongInf[n, 1]
            #
            BelongInf_tosplit, Clusters_tosplit = K_Means(ClusterData_tosplit, 2)
            SSE_tosplit = np.sum(BelongInf_tosplit[:, 1])
            #
            if SSE_tosplit + SSE_notsplit < LowestSSE:
                BestToSplit_ClusterID = i
                BestToSplit_BelongInf = BelongInf_tosplit.copy()
                BestToSplit_Clusters = Clusters_tosplit.copy()
                LowestSSE = SSE_tosplit + SSE_notsplit
        #
        count = 0
        for j in range(N):
            if BelongInf[j, 0] == BestToSplit_ClusterID:
                if BestToSplit_BelongInf[count, 0] == 0:
                    BelongInf[j, 0] = BestToSplit_ClusterID
                    BelongInf[j, 1] = BestToSplit_BelongInf[count, 1]
                elif BestToSplit_BelongInf[count, 0] == 1:
                    BelongInf[j, 0] = k
                    BelongInf[j, 1] = BestToSplit_BelongInf[count, 1]
                count += 1
        #
        Clusters[BestToSplit_ClusterID, :] = BestToSplit_Clusters[0, :]
        Clusters[k, :] = BestToSplit_Clusters[1, :]
        #
        k += 1
    #
    BelongInf, Clusters = K_Means(Data, K, BelongInf, Clusters)
    #
    print(Clusters)
    return(BelongInf, Clusters)
################################################################################
def main():
    #
    '''
    AP_u, AP_v, AP_w, u_AP, v_AP, w_AP = ReadFile('AP_local_Ind.dat')
    #
    params_u = np.ones(2)
    params_v = np.ones(2)
    params_w = np.ones(3)
    paramsLims_u =np.array([[0.0, 2.0], [-100.0, 100.0]])
    paramsLims_v =np.array([[0.0, 2.0], [-100.0, 100.0]])
    paramsLims_w =np.array([[0.0, 2.0], [-100.0, 100.0], [0.0, 100.0]])
    #
    params_u, params_Loop_u, Error_Loop_u = Fitting2D(AP_u, u_AP, Maxwellian_T, params_u, paramsLims_u, 100000, 'u')
    params_v, params_Loop_v, Error_Loop_v = Fitting2D(AP_v, v_AP, Maxwellian_T, params_v, paramsLims_v, 100000, 'v')
    params_w, params_Loop_w, Error_Loop_w = Fitting2D(AP_w, w_AP, Maxwellian_N, params_w, paramsLims_w, 100000, 'w')
    #
    Fitting2D_Plot(AP_u, AP_v, AP_w, u_AP, v_AP, w_AP, params_u, params_v, params_w, Maxwellian_T, Maxwellian_T, Maxwellian_N)
    #
    Fitting2D_Loops_dat(params_Loop_u, Error_Loop_u, 'u')
    Fitting2D_Loops_Plot(params_Loop_u, Error_Loop_u, 'u')
    Fitting2D_Loops_dat(params_Loop_v, Error_Loop_v, 'v')
    Fitting2D_Loops_Plot(params_Loop_v, Error_Loop_v, 'v')
    Fitting2D_Loops_dat(params_Loop_w, Error_Loop_w, 'w')
    Fitting2D_Loops_Plot(params_Loop_w, Error_Loop_w, 'w')
    '''
    #
    #Data = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [0.8, 2.2], [2.0, 1.0], [2.1, 1.1], [1.9, 0.9], [2.2, 0.8]])
    K = 4
    Dim = 3
    N =100
    #
    Data = np.zeros((N, Dim))
    TempData = np.random.normal((np.random.random() - 0.5) * 14, 1, (N, 1))
    for d in range(Dim - 1):
        TempData = np.append(TempData, np.random.normal((np.random.random() - 0.5) * 14, 1, (N, 1)), axis = 1)
    Data[:, :] = TempData[:, :]
    for k in range(K - 1):
        TempData = np.random.normal((np.random.random() - 0.5) * 14, 1, (N, 1))
        for d in range(Dim - 1):
            TempData = np.append(TempData, np.random.normal((np.random.random() - 0.5) * 14, 1, (N, 1)), axis = 1)
        Data = np.append(Data, TempData, axis = 0)
    '''
    #
    BelongInf, Clusters = K_Means(Data, K)
    #
    fig, ax = mpl.subplots(figsize = (6.4, 6.4))
    N, Dim = Data.shape
    markers = ['o', 'v', '^', 's', 'P', '*', 'D', 'X']
    colours = ['r', 'b', 'g', 'c', 'm', 'y', 'tomato', 'skyblue']
    for i in range(N):
        ax.plot(Data[i, 0], Data[i, 1], color = colours[int(BelongInf[i, 0])], marker = markers[int(BelongInf[i, 0])], markersize = 3)
    ax.plot(Clusters[:, 1], Clusters[:, 2], 'kx', markersize = 9)
    ax.set_xlim(-11.0, 11.0)
    ax.set_ylim(-11.0, 11.0)
    mpl.savefig('K_Means.png', dpi = 600)
    fig.show()
    #
    BelongInf, Clusters = Bi_K_Means(Data, K)
    #
    fig, ax = mpl.subplots(figsize = (6.4, 6.4))
    N, Dim = Data.shape
    markers = ['o', 'v', '^', 's', 'P', '*', 'D', 'X']
    colours = ['r', 'b', 'g', 'c', 'm', 'y', 'tomato', 'skyblue']
    for i in range(N):
        ax.plot(Data[i, 0], Data[i, 1], color = colours[int(BelongInf[i, 0])], marker = markers[int(BelongInf[i, 0])], markersize = 3)
    ax.plot(Clusters[:, 1], Clusters[:, 2], 'kx', markersize = 9)
    ax.set_xlim(-11.0, 11.0)
    ax.set_ylim(-11.0, 11.0)
    mpl.savefig('Bi_K_Means.png', dpi = 600)
    fig.show()
    '''
    #
    BelongInf, Clusters = Bi_K_Means_Wong(Data, K)
    #
    #fig, ax = mpl.subplots(figsize = (6.4, 6.4, 6.4))
    fig = mpl.figure()
    ax = fig.add_subplot(111, projection = '3d')
    N, Dim = Data.shape
    markers = ['o', 'v', '^', 's', 'P', '*', 'D', 'X']
    colours = ['r', 'b', 'g', 'c', 'm', 'y', 'tomato', 'skyblue']
    for i in range(N):
        ax.scatter(Data[i, 0], Data[i, 1], Data[i, 2], s = 5, c = colours[int(BelongInf[i, 0])], marker = markers[int(BelongInf[i, 0])])
    ax.scatter(Clusters[:, 1], Clusters[:, 2], Clusters[:, 3], s = 15, c = 'k', marker = 'x')
    ax.set_xlim(-9.0, 9.0)
    ax.set_ylim(-9.0, 9.0)
    ax.set_zlim(-9.0, 9.0)
    mpl.savefig('Bi_K_Means_Wong.png', dpi = 600)
    fig.show()

################################################################################
if __name__ == '__main__':
    #
    main()
