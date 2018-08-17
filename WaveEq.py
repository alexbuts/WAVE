from scipy import interpolate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sin,cos,exp
import numpy as np
class MixCond():
    @staticmethod
    def phi(t):#neiman du(T,t)/dn=2t T-boundaries of rectangle
        return 2*t
    @staticmethod
    def f(x,y,t):
        return exp(t)*sin(x)*sin(y)
    @staticmethod
    def g1(x,y):#u(0,x)=g1(x)
        return cos(x)*cos(y)
    @staticmethod
    def g2(x,y):#du(0,x)/dt=g2(x)
        return sin(x)*cos(y)
class PDE(MixCond):#d2u/dt2=v*(d2u/dx2+d2u/dy2)+f(x,y,t);
    def __init__(self,_v,_x,_y,_t):#choose speed, rectangle [0;X]x[0;Y] and period of time T
        self.T=_t
        self.X=_x
        self.Y = _y
        assert _v!=0,"its not a wave equation"
        self.v=_v
    #make a net for finite difference method
    Net = {}
    def setNetandWeights(self,h1,h2,t0,wg):
        def points(Rend,step):
            line=[]
            Lend=0
            while Lend<=Rend:
                line.append(Lend)
                Lend+=step
            return line
        assert (self.v*(t0**2)/(h1**2+h2**2))*(1-4*wg)<=1,"doesn't violate  Courant condition"
        self.Net["stepX"] = h1
        self.Net["stepY"] = h2
        self.Net["stepTime"] = t0
        self.Net["Xlist"] = points(self.X,h1)
        self.Net["Ylist"] = points(self.Y,h2)
        self.Net["Timelist"] = points(self.T,t0)
        self.Net["weight"] = wg
    def Solution(self,_x,_y,t):
        H1,H2,T0,Wgh=self.Net["stepX"],self.Net["stepY"],self.Net["stepTime"],self.Net["weight"]
        Xc,Yc,Tc=self.Net["Xlist"],self.Net["Ylist"],self.Net["Timelist"]
        u=[]#index for SLAE
        A=[[0*i for i in range(len(Tc)*len(Xc)*len(Yc))] for j in range(len(Tc)*len(Xc)*len(Yc))]#
        b=[]
        z=0 #strings of A system
        #Make a SLAE
        for i in range(len(Xc)):
            for j in range(len(Yc)):
                for k in range(len(Tc)):
                    u.append((Xc[i],Yc[j],Tc[k]))
        for i in range(len(Xc)): #start condition for zero time level
            for j in range(len(Yc)):
                b.append(PDE.g1(Xc[i],Yc[j]))
                A[z][len(Tc) * j + len(Tc) * len(Yc) * i] = 1
                z+=1
        for i in range(len(Xc)):#start condition for first time level
            for j in range(len(Yc)):
                b.append(T0*PDE.g2(Xc[i],Yc[j]))
                A[z][len(Tc) * j + len(Tc) * len(Yc) * i] = -1
                A[z][1+len(Tc) * j + len(Tc) * len(Yc) * i] = 1
                z+=1
        for j in range(1,len(Yc)-1): #Neumann boundary condition  Y axis
            for k in range(2,len(Tc)):
                A[z][k + len(Tc) * j + len(Tc) * len(Yc) * (len(Xc)-2)] = 1/H1
                A[z][k + len(Tc) * j + len(Tc) * len(Yc) * (len(Xc)-1)] = -1/H1
                b.append(PDE.phi(Tc[k]))
                z+=1
                A[z][k + len(Tc) * j + len(Tc) * len(Yc) * 1] =1/H1
                A[z][k + len(Tc) * j + len(Tc) * len(Yc) * 0] =-1/H1
                b.append(PDE.phi(Tc[k]))
                z += 1
        for i in range(len(Xc)):# Neumann boundary condition  X axis
            for k in range(2,len(Tc)):
                A[z][k + len(Tc) * (len(Yc)-2) + len(Tc) * len(Yc) * i] = 1/H2
                A[z][k + len(Tc) * (len(Yc)-1) + len(Tc) * len(Yc) * i] = -1/H2
                b.append(PDE.phi(Tc[k]))
                z+=1
                A[z][k + len(Tc) * 1 + len(Tc) * len(Yc) * i] =1/H2
                A[z][k + len(Tc) * 0 + len(Tc) * len(Yc) * i] =-1/H2
                b.append(PDE.phi(Tc[k]))
                z += 1
        for i in range(1,len(Xc)-1):# Crank–Nicolson method
            for j in range(1,len(Yc)-1):
                for k in range(1,len(Tc)-1):
                    b.append(PDE.f(Xc[i],Yc[j],Tc[k]))
                    A[z][k+len(Tc)*j+len(Tc)*len(Yc)*i]=-2/(T0**2)+self.v * 2*(1-2*Wgh)*(1/(H1**2)+1/(H2**2))
                    A[z][k + 1 + len(Tc) * j + len(Tc) * len(Yc) * i] = A[z][k - 1 + len(Tc) * j + len(Tc) * len(Yc) * i]=1/(T0**2) + self.v *2*Wgh*(1/(H1**2) + 1/(H2 ** 2))
                    A[z][k  + len(Tc) * j + len(Tc) * len(Yc) * (i+1)]= -self.v *(1-2*Wgh)/(H1**2)
                    A[z][k - 1  + len(Tc) * j + len(Tc) * len(Yc) * (i + 1)] = -self.v *Wgh/(H1**2)
                    A[z][k + 1 + len(Tc) * j + len(Tc) * len(Yc) * (i + 1)] = -self.v *Wgh/(H1**2)
                    A[z][k + len(Tc) * j + len(Tc) * len(Yc) * (i - 1)] = -self.v *(1-2*Wgh)/(H1**2)
                    A[z][k + 1 + len(Tc) * j + len(Tc) * len(Yc) * (i - 1)] = -self.v *Wgh / (H1 ** 2)
                    A[z][k - 1 + len(Tc) * j + len(Tc) * len(Yc) * (i - 1)] = -self.v *Wgh / (H1 ** 2)
                    A[z][k + len(Tc) * (j+1) + len(Tc) * len(Yc) * i ] = -self.v *(1 - 2 * Wgh) / (H2 ** 2)
                    A[z][k - 1 + len(Tc) * (j+1) + len(Tc) * len(Yc) *i ] = -self.v *Wgh / (H2 ** 2)
                    A[z][k + 1 + len(Tc) * (j+1) + len(Tc) * len(Yc) *i ] = -self.v *Wgh / (H2 ** 2)
                    A[z][k + len(Tc) * (j-1) + len(Tc) * len(Yc) * i ] = -self.v *(1 - 2 * Wgh) / (H2 ** 2)
                    A[z][k + 1 + len(Tc) * (j-1) + len(Tc) * len(Yc) *i] = -self.v *Wgh / (H2 ** 2)
                    A[z][k - 1 + len(Tc) * (j-1) + len(Tc) * len(Yc) * i ] = -self.v *Wgh / (H2 ** 2)
                    z+=1
        A=csr_matrix(A)
        res=spsolve(A,np.array(b))#solve system and finding nodes
        return np.float64(interpolate.griddata(u,res,(_x,_y,t)))#interpolation
    def XoTPr(self,count):
        X = np.arange(0, self.X, self.X /count)
        Y = np.arange(0, self.T, self.T / count)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([np.array([self.Solution(i,0, j) for i in np.arange(0, self.X, 0.05)]) for j in np.arange(0, self.T, 0.1)])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("T Axis")
        ax.plot_surface(X, Y,Z)
    def YoTPr(self,count):
        X = np.arange(0, self.Y, self.Y / count)
        Y = np.arange(0, self.T, self.T /count)
        X, Y = np.meshgrid(X, Y)
        Z = np.array([np.array([self.Solution(0,i, j) for i in np.arange(0, self.Y, 0.1)]) for j in np.arange(0, self.T, 0.1)])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("Y Axis")
        ax.set_ylabel("T Axis")
        ax.plot_surface(X, Y, Z,color='green')
if __name__=="__main__":
    a=PDE(0.25,1,2,2)
    a.setNetandWeights(0.2,0.4,0.4,0.5)
    a.XoTPr(20)
    a.YoTPr(20)
    plt.show()


