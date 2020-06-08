import numpy as np
import numba as nb

################################################################################
@nb.jit(nopython = True, nogil = True)
def Maxwell(u):
    #
    f = np.sqrt(1 / np.pi) * np.exp(- u ** 2)
    #f = 2 * abs(u) * np.exp(- u ** 2)
    #
    return(f)

################################################################################
@nb.jit(nopython = True, nogil = True)
def CLL_R(ur, ui):
    #
    sigma = 0.5
    R = 1 / np.sqrt(np.pi * sigma * (2 - sigma)) * np.exp(- (ur - (1 - sigma) * ui) ** 2 / (sigma * (2 - sigma)))
    #
    return(R)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Initial():
    #
    N = 100
    ur = np.zeros(N + 1)
    ui = np.zeros(N + 1)
    Low0 = -5.0
    Low1 = 0.0
    High0 = 0.0
    High1 = 5.0
    for i in range(len(ur)):
        ur[i] = Low0 + (High1 - Low0) * i / N
        ui[i] = Low0 + (High1 - Low0) * i / N
        #vi[i] = Low0 + (High1 - Low0) * i / N
        #wi[i] = Low0 + (High0 - Low0) * i / N
    #
    return(ur, ui)

################################################################################
def main():
    #
    ur, ui = Initial()
    Sum = 0.0
    for i in range(len(ui)):
        Sum += CLL_R(ur[50], ui[i]) * Maxwell(ui[i]) * (ui[1] - ui[0])
    print(Sum)

################################################################################
if __name__ == '__main__':
    #
    main()
