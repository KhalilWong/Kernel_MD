import numpy as np
import matplotlib.pyplot as mpl
import random

################################################################################
def Function(x):
    f = (x - 1.0) ** 2 + 2.0 + random.random()
    return f

################################################################################
def Model(x, p):
    f = 0.0
    for i in range(len(p)):
        f += p[i] * x ** i
    return f

################################################################################
def NAG(x, y_target, Nor_Type = ''):
    N = len(x)
    y_prediction = np.zeros(N)
    p = np.zeros(7)
    v = np.zeros(len(p))
    #
    mu = 0.999
    dp = 1e-9
    lr = 1e-6
    lam = 1.0
    Loops = 10000
    Error_Loops = np.zeros(Loops)
    for i in range(Loops):
        print(i)
        Error = 0.0
        Error_p = np.zeros(len(p))
        for j in range(N):
            y_prediction[j] = Model(x[j], p)
            Error += (y_prediction[j] - y_target[j]) ** 2
            for k in range(len(p)):
                p[k] += dp
                y_prediction_p = Model(x[j], p)
                p[k] -= dp
                Error_p[k] += (y_prediction_p - y_target[j]) ** 2
                #print(Error_p[k])
        #
        if Nor_Type == '':
            pass
        elif Nor_Type == 'L1':
            for m in range(len(p)):
                Error += lam * np.abs(p[m])
                for j in range(len(p)):
                    if j == m:
                        Error_p[j] += lam * np.abs(p[m] + dp)
                    else:
                        Error_p[j] += lam * np.abs(p[m])
        elif Nor_Type == 'L2':
            for m in range(len(p)):
                Error += 0.5 * lam * p[m] ** 2
                for j in range(len(p)):
                    if j == m:
                        Error_p[j] += 0.5 * lam * (p[m] + dp) ** 2
                    else:
                        Error_p[j] += 0.5 * lam * p[m] ** 2
        #
        for j in range(len(p)):
            dErr_dp = (Error_p[j] - Error) / dp
            pre_v = v[j]
            v[j] = mu * v[j]
            v[j] += - lr * dErr_dp
            p[j] += v[j] + mu * (v[j] - pre_v)
        Error_Loops[i] = Error
    #
    return(y_prediction, p, Error_Loops)

################################################################################
def main():
    x = np.arange(-2.0, 2.0, 0.1)
    y_target = np.zeros(len(x))
    for i in range(len(x)):
        y_target[i] = Function(x[i])
    #
    y_prediction, p, Error_Loops = NAG(x, y_target)
    y_prediction_L1, p_L1, Error_Loops_L1 = NAG(x, y_target, 'L1')
    y_prediction_L2, p_L2, Error_Loops_L2 = NAG(x, y_target, 'L2')
    #
    fig, ax = mpl.subplots()
    ax.plot(x, y_target, 'r', label = 'Target: $f = x^{2} - 2x + 3 + noise$')
    ax.plot(x, y_prediction, 'b', label = 'Prediction: $f = x^{' + str(round(p[2], 3)) + '} - ' + str(round(p[1], 3)) + 'x + ' + str(round(p[0], 3)) + '$')
    ax.plot(x, y_prediction_L1, 'g', label = 'Prediction_L1: $f = x^{' + str(round(p_L1[2], 3)) + '} - ' + str(round(p_L1[1], 3)) + 'x + ' + str(round(p_L1[0], 3)) + '$')
    ax.plot(x, y_prediction_L2, 'm', label = 'Prediction_L2: $f = x^{' + str(round(p_L2[2], 3)) + '} - ' + str(round(p_L2[1], 3)) + 'x + ' + str(round(p_L2[0], 3)) + '$')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0.0, 11.0)
    ax.legend(loc = 'upper right', fontsize = 'small', frameon = False)
    mpl.savefig('tar-pre.png', dpi = 600)
    mpl.show()
    fig, ax = mpl.subplots()
    ax.plot(p, 'b', label = 'P: ' + str(round(p[0], 3)) + '; ' + str(round(p[1], 3)) + '; ' + str(round(p[2], 3)) + '; ' + str(round(p[3], 3)) + '; ' + str(round(p[4], 3)) + '; ' + str(round(p[5], 3)) + '; ' + str(round(p[6], 3)))
    ax.plot(p_L1, 'g', label = 'P_L1: ' + str(round(p_L1[0], 3)) + '; ' + str(round(p_L1[1], 3)) + '; ' + str(round(p_L1[2], 3)) + '; ' + str(round(p_L1[3], 3)) + '; ' + str(round(p_L1[4], 3)) + '; ' + str(round(p_L1[5], 3)) + '; ' + str(round(p_L1[6], 3)))
    ax.plot(p_L2, 'm', label = 'P_L2: ' + str(round(p_L2[0], 3)) + '; ' + str(round(p_L2[1], 3)) + '; ' + str(round(p_L2[2], 3)) + '; ' + str(round(p_L2[3], 3)) + '; ' + str(round(p_L2[4], 3)) + '; ' + str(round(p_L2[5], 3)) + '; ' + str(round(p_L2[6], 3)))
    ax.legend(loc = 'upper right', fontsize = 'small', frameon = False)
    mpl.savefig('p.png', dpi = 600)
    mpl.show()
    fig, ax = mpl.subplots()
    ax.plot(Error_Loops, 'b', label = 'Origin')
    ax.plot(Error_Loops_L1, 'g', label = 'L1')
    ax.plot(Error_Loops_L2, 'm', label = 'L2')
    ax.legend(loc = 'upper right', fontsize = 'small', frameon = False)
    mpl.savefig('Error.png', dpi = 600)
    mpl.show()

################################################################################
if __name__ == '__main__':
    main()
