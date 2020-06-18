import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import numba as nb
import matplotlib.pyplot as mpl

################################################################################
def ReadFile1(FileName):
    #
    with open(FileName, 'r') as In:
        Data = In.readlines()
        InData = Data[1:]
        ID = []
        Tt = []
        #CN = []
        #TmVt = []
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
            (id, tt, x, y, z, vx, vy, vz, fx, fy, fz, fvx, fvy, fvz, ad) = pdata.split('\t', 14)
            ID.append(int(id))
            Tt.append(int(tt))
            #CN.append(int(cn))
            #TmVt.append(int(tmvt))
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
    #CN = np.array(CN)
    #TmVt = np.array(TmVt)
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
    return(ID, Tt, X, Y, Z, VX, VY, VZ, FX, FY, FZ, FVX, FVY, FVZ, Adsorbed)

################################################################################
def ReadFile2(FileName):
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
class Net(nn.Module):
    #
    def __init__(self):
        #
        super().__init__()
        self.conv1 = nn.Conv1d(3, 6, 1000)
        self.conv2 = nn.Conv1d(6, 9, 1000)
        self.conv3 = nn.Conv1d(9, 12, 1000)
        self.conv4 = nn.Conv1d(12, 15, 1000)
        #
        self.fc1 = nn.Linear(79695, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 31 * 3)
        #
        #self.conv1 = nn.Conv1d(1, 3, 5)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 2)
        x = F.max_pool1d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

################################################################################
#@nb.jit(nopython = True, nogil = True)
def Reflection_Distribution(UnA_N, X, Adsorbed, I, XL = 0.0, XH = 0.0):
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
    f = np.zeros(I + 1)                                                         #总反射
    for i in range(I + 1):
        count = 0
        for n in range(len(X)):
            if Adsorbed[n] != 1 and XLow + (i - 0.5) * dX <= X[n] < XHigh - (I - i - 0.5) * dX:
                count += 1
        x[i] = (XLow + XHigh + (2 * i - I) * dX) / 2
        f[i] = count / (UnA_N * dX)
    #
    return(x, f)

################################################################################
#@nb.jit(nopython = True, nogil = True)
def Learning_Plot(ur, vr, wr, output, Type):
    #
    d, N = output.shape
    #
    for i in range(d - 1):
        fig, ax = mpl.subplots()
        #levels = np.arange(0.0, 0.75, 0.05)
        #mpl.contourf(ur, ui, output[:, :, i], levels)
        #mpl.colorbar()
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(0.0, 0.7)
        if i == 0:
            ax.plot(ur, output[i, :], 'ro', markersize = 3)
            ax.set_xlabel('$u_{r}$')
            ax.set_ylabel('$f$')
            #ax.set_title('p0 = %f;p1 = %f'%(p[0], p[1]))
            mpl.savefig('CNN_u_'+ Type +'.png', dpi = 600)
        else:
            ax.plot(vr, output[i, :], 'ro', markersize = 3)
            ax.set_xlabel('$v_{r}$')
            ax.set_ylabel('$f$')
            #ax.set_title('p0 = %f;p1 = %f'%(p[0], p[1]))
            mpl.savefig('CNN_v_'+ Type +'.png', dpi = 600)
        #mpl.show()
        mpl.close()
    #
    fig, ax = mpl.subplots()
    #levels = np.arange(0.0, 1.05, 0.05)
    #mpl.contourf(ui[:, 1], wr[:, 1], output[:, :, 2], levels)
    #mpl.colorbar()
    ax.plot(wr, output[2, :], 'ro', markersize = 3)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('$w_{r}$')
    ax.set_ylabel('$f$')
    #ax.set_title('p0 = %f;p1 = %f'%(p[0], p[1]))
    mpl.savefig('CNN_w_'+ Type +'.png', dpi = 600)
    #mpl.show()
    mpl.close()

################################################################################
def main():
    #
    '''
    FileName = 'Incident_Reflection.data'
    ID, Tt, X, Y, Z, VX, VY, VZ, FX, FY, FZ, FVX, FVY, FVZ, Adsorbed = ReadFile1(FileName)
    UnA_N = 0
    for i in range(len(Adsorbed)):
        if Adsorbed[i] == 0:
            UnA_N += 1
    #
    Training_np = np.concatenate((VX, VY, VZ), axis = 0)
    Training = torch.from_numpy(Training_np)
    Training = Training.view(1, 3, -1)
    #
    I = 30
    ur, Targetx = Reflection_Distribution(UnA_N, FVX, Adsorbed, I, -2.0, 2.0)
    vr, Targety = Reflection_Distribution(UnA_N, FVY, Adsorbed, I, -2.0, 2.0)
    wr, Targetz = Reflection_Distribution(UnA_N, FVZ, Adsorbed, I, 0.0, 2.0)
    Target_np = np.concatenate((Targetx, Targety, Targetz), axis = 0).reshape((3, 31))
    Learning_Plot(ur, vr, wr, Target_np, 'Target')
    Target = torch.from_numpy(Target_np)
    Target = Target.view(1, -1)
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Net()
    net.to(device)
    Training_d = Training.to(device, dtype=torch.float)
    Target_d = Target.to(device, dtype=torch.float)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    #
    for i in range(1000):
        print(i)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        Prediction = net(Training_d)
        loss = criterion(Prediction, Target_d)
        print(loss.item())
        loss.backward()
        #print(torch.cuda.memory_summary())
        optimizer.step()
    #
    PATH = './distribution_net.pth'
    torch.save(net.state_dict(), PATH)
    output = Prediction.to("cpu", torch.double)
    output = output.view(3, 31)
    output_np = output.detach().numpy()
    Learning_Plot(ur, vr, wr, output_np, 'Prediction')
    #
    '''
    #
    FileName = 'Incident_Reflection_Test.data'
    ID, Tt, CN, TmVt, X, Y, Z, VX, VY, VZ, FX, FY, FZ, FVX, FVY, FVZ, Adsorbed = ReadFile2(FileName)
    UnA_N = 0
    for i in range(len(Adsorbed)):
        if Adsorbed[i] == 0:
            UnA_N += 1
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    Test_np = np.concatenate((VX, VY, VZ), axis = 0)
    Test_np[0:100000] = 1.0
    Test_np[100000:200000] = 1.0
    Test_np[200000:300000] = np.sqrt(2.0)
    Test = torch.from_numpy(Test_np)
    Test = Test.view(1, 3, -1)
    Test_d = Test.to(device, dtype=torch.float)
    I = 30
    ur, Targetx = Reflection_Distribution(UnA_N, FVX, Adsorbed, I, -2.0, 2.0)
    vr, Targety = Reflection_Distribution(UnA_N, FVY, Adsorbed, I, -2.0, 2.0)
    wr, Targetz = Reflection_Distribution(UnA_N, FVZ, Adsorbed, I, 0.0, 2.0)
    #
    with torch.no_grad():
        net = Net()
        net.load_state_dict(torch.load('./distribution_net.pth'))
        net.to(device)
        test_output = net(Test_d)
        test_output = test_output.to("cpu", torch.double)
        test_output = test_output.view(3, 31)
        test_output_np = test_output.detach().numpy()
        Learning_Plot(ur, vr, wr, test_output_np, 'Test')

################################################################################
if __name__ == '__main__':
    main()
