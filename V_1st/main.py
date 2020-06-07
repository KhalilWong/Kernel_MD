from parameters import Parameters
from Progress_Bar import Progress_Bar
import functions as df
import os
from numpy import *
from matplotlib.pyplot import *

global Pt_Pos
global Pt_Vel
global Pt_Acc
global Ar_Pos
global Ar_Vel
global Ar_Acc

Pars=Parameters()
(Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc,timestep)=df.Initialization(Pars)
df.Dump(Pt_Pos,Ar_Pos,Pars,timestep)
Bar=Progress_Bar(Pars.Tt,'MD')
while(Pars.state):
    (Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc)=df.Verlet(Ar_Pos,Ar_Vel,Ar_Acc,Pt_Pos,Pt_Vel,Pt_Acc,Pars)
    (Pt_Vel)=df.rescale_T(Pt_Vel,Pars)
    timestep+=1
    df.Exit(Pt_Pos,Ar_Pos,Pars,timestep)
    Bar.show()
Bar.close()
df.V_plot(Pars,Pt_Vel)
