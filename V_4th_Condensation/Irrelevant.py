import numpy as np
import numba as nb
import random
import matplotlib.pyplot as mpl

################################################################################
@nb.jit(nopython = True, nogil = True)
def CLL_R(wi, wr, ui, vi, p):
    #
    alpha = p[0] * np.exp(- p[1] * ui ** 2 - p[1] * vi ** 2)
    N = 200
    R = 0.0
    for i in range(N):
        R += 2 * wr / alpha * np.exp(- (wr ** 2 + (1 - alpha) * wi ** 2 - 2 * np.sqrt(1 - alpha) * wr * wi * np.cos((i + 0.5) / N * np.pi)) / alpha) / N
    #
    return(R)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Irrelevant(Model, ui, wr, vi, wi, fui, fvi, fwi, GasTmpV, WallTmpV, lt):
    #
    Mod_Rfu = np.zeros((len(ui), len(wr)))
    #
    for i in range(len(ui)):
        if fui[i] != 0:
            dfui = fui[i]
            for j in range(len(wr)):                                            #vr, wr
                for k in range(len(vi)):
                    for m in range(len(wi)):
                        Mod_Rfu[i, j] += Model(wi[m, 1] * GasTmpV / WallTmpV, wr[j, 1] * GasTmpV / WallTmpV, ui[i, 1] * GasTmpV / WallTmpV, vi[k, 1] * GasTmpV / WallTmpV, lt) * GasTmpV / WallTmpV * dfui * fvi[k] * fwi[m] * (vi[k, 2] - vi[k, 0]) * (wi[m, 2] - wi[m, 0])
            Mod_Rfu[i, :] /= dfui
    #
    return(Mod_Rfu)

################################################################################
def main():
    #
    N = 50
    ui = np.zeros((N + 1, 3))
    wr = np.zeros((N + 1, 3))
    vi = np.zeros((N + 1, 3))
    wi = np.zeros((N + 1, 3))
    fui = np.zeros(N + 1)
    fvi = np.zeros(N + 1)
    fwi = np.zeros(N + 1)
    p = np.zeros(2)
    #p[0] = random.random()
    #p[1] = 2 * random.random()
    p[0] = 1.0
    p[1] = 0.0
    GasTmpV = 1.0
    WallTmpV = 1.0
    #
    Low0 = - 5.0
    Low1 = 0.0
    High0 = 0.0
    High1 = 5.0
    for i in range(N + 1):
        ui[i, 0] = Low0 + (High1 - Low0) * (i - 0.5) / N
        wr[i, 0] = Low1 + (High1 - Low1) * (i - 0.5) / N
        vi[i, 0] = Low0 + (High1 - Low0) * (i - 0.5) / N
        wi[i, 0] = Low0 + (High0 - Low0) * (i - 0.5) / N
        ui[i, 1] = Low0 + (High1 - Low0) * i / N
        wr[i, 1] = Low1 + (High1 - Low1) * i / N
        vi[i, 1] = Low0 + (High1 - Low0) * i / N
        wi[i, 1] = Low0 + (High0 - Low0) * i / N
        ui[i, 2] = Low0 + (High1 - Low0) * (i + 0.5) / N
        wr[i, 2] = Low1 + (High1 - Low1) * (i + 0.5) / N
        vi[i, 2] = Low0 + (High1 - Low0) * (i + 0.5) / N
        wi[i, 2] = Low0 + (High0 - Low0) * (i + 0.5) / N
        #
        fui[i] = np.sqrt(1 / np.pi) * np.exp(- ui[i, 1] ** 2)
        fvi[i] = np.sqrt(1 / np.pi) * np.exp(- vi[i, 1] ** 2)
        fwi[i] = 2 * abs(wi[i, 1]) * np.exp(- wi[i, 1] ** 2)
    #
    print('Initial Over')
    Mod_R = Irrelevant(CLL_R, ui, wr, vi, wi, fui, fvi, fwi, GasTmpV, WallTmpV, p)
    fig, ax = mpl.subplots()
    #mpl.plot(ui[:, 1], fui)
    levels = np.arange(0.0, 1.05, 0.05)
    mpl.contourf(ui[:, 1], wr[:, 1], Mod_R.T, levels)
    mpl.colorbar()
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel('$u_{i}$')
    ax.set_ylabel('$w_{r}$')
    ax.set_title('p0 = %f;p1 = %f'%(p[0], p[1]))
    mpl.savefig('Irrelevant_(p0 = %f;p1 = %f).png'%(p[0], p[1]), dpi = 600)
    mpl.show()
    #mpl.close()

################################################################################
if __name__ == '__main__':
    #
    main()
