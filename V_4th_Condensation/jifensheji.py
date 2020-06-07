import numpy as np

################################################################################
@nb.jit(nopython = True, nogil = True)
def Maxwell(ur, ui):
    #
    sigma = 0.5
    #f = np.sqrt(1 / np.pi) * np.exp(- u ** 2)
    f = 1 / np.sqrt(np.pi * sigma * (2 - sigma)) * np.exp(- (ur - (1 - sigma) * ui) ** 2 / (sigma * (2 - sigma)))
    #f = 2 * abs(u) * np.exp(- u ** 2)
    #
    return(f)

################################################################################
@nb.jit(nopython = True, nogil = True)
def Initial(ur, ui):
    #
    Low0 = -5.0
    Low1 = 0.0
    High0 = 0.0
    High1 = 5.0
    for i in range(len(ur)):
        ur[i] = Low0 + (High1 - Low0) * (i - 0.5) / N
        ui[i] = Low0 + (High1 - Low0) * (i - 0.5) / N
        #vi[i] = Low0 + (High1 - Low0) * (i - 0.5) / N
        #wi[i] = Low0 + (High0 - Low0) * (i - 0.5) / N
    #
    return(ur, ui)

################################################################################
def main():
    #
    N = 50
    ur = np.zeros(N + 1)
    ui = np.zeros(N + 1)
    ur, ui = Initial(ur, ui)

################################################################################
if __name__ == '__main__':
    #
    main()
