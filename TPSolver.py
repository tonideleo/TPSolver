from re import A
from turtle import color
import numpy as np
import cupy as cp
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg as cpl
from regex import W
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

import time
from numba import cuda
from progress.bar import Bar # (pip install progress)
from cpuinfo import get_cpu_info
import os

clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')



import warnings
# warnings.filterwarnings("error")
# warnings.filterwarnings('ignore')



class TPSolver:
    nu = 0.001      # Kinematic Viscosity
    rho = 1.225     # Fluid Density
    
    nx = 10         # Number of Grid Points in x
    ny = 10         # Number of Grid Points in y
    
    Lx = 1          # Domain Length in x
    Ly = 1          # Domain Length in y

    dt = 0.001      # Time Step Size
    tf = 1          # Simulation Time
    nsteps = 0      # Number of Time Steps
    t = 0           # Current Simulation Time
    
    # INDICES  
    imin = imax = jmin = jmax = 0
    
    # Miscellaneous
    __type = np.float64
    flagGPU = False
    linelenght = 70
    plot_frequency = 100
    sig_figs = 3
    
    # Matrices
    x = y = xm = ym = p = us = vs = R = u = v = L = L_sp = []
    
    # Initial Velocities
    u_bot = u_top = v_left = v_right = 0
    
    # Mesh size
    dx = dy = dxi = dxy = 0
    
    debug = flagTimeStatistics = flagPlot = False
    verbose = True
    
    def __init__(self, license = True, GPUmode = False):
        if license:
            self.licenseDisclaimer()
        if not GPUmode:
            return
        self.flagGPU = GPUmode

    def enableGPU(self,mode):
        self.flagGPU = mode

    def setVerbose(self,val):
        self.verbose = val
            
    def plotEveryNTimeSteps(self,val):
        self.plot_frequency = val
        self.flagPlot = True
    
    def setDebug(self,val):
        self.debug = val
    def printLine(self,char = '-'):
        s = char
        print(s * self.linelenght)
    def printTextOnLine(self,text,char = '-'):
        L = len(text)
        LP = (self.linelenght - L)//2        
        if LP % 2 == 0:
            print(char*LP,text,char*LP)
        else:
            print(char*(LP - 1),text,char*(LP))
    def licenseDisclaimer(self):
        self.printTextOnLine('License Disclaimer','=')
        print('TPSolver: 2D Navier-Stokes Solver for GPU')
        print('This program is free software: you can redistribute it and/or modify')
        print('it under the terms of the GNU Affero General Public License as published')
        print('by the Free Software Foundation, either version 3 of the License.')
        print('This program is distributed in the hope that it will be useful,')
        print('but WITHOUT ANY WARRANTY; without even the implied warranty of')
        print('MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the')
        print('GNU Affero General Public License for more details.')
        self.printLine('=')
        print('')
            
    def printHeaderAndOptions(self):
        self.printTextOnLine('TPSolver','=')
        # print('Options selected:')
        device = 'GPU' if self.flagGPU else 'CPU'
        print('Device Mode: ', device)
        if device == 'GPU':
            cc_cores_per_SM_dict = {
                (2,0) : 32,
                (2,1) : 48,
                (3,0) : 192,
                (3,5) : 192,
                (3,7) : 192,
                (5,0) : 128,
                (5,2) : 128,
                (6,0) : 64,
                (6,1) : 128,
                (7,0) : 64,
                (7,5) : 64,
                (8,0) : 64,
                (8,6) : 128}
            gpu = cuda.get_current_device()
            str = getattr(gpu,'name')
            print('GPU Model: ',str.decode("utf-8"))
            my_sms = getattr(gpu, 'MULTIPROCESSOR_COUNT')
            my_cc = gpu.compute_capability
            cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
            total_cores = cores_per_sm*my_sms
            print("GPU Compute Capability: " , my_cc)
            print("GPU Total Number of SMs: " , my_sms)
            print("GPU Total Cores: " , total_cores)
            
        cpu = get_cpu_info()
        print('CPU Model: ',cpu['brand_raw'])
        print('CPU Total Cores: ',cpu['count'])
        print('Debug Mode: ', self.debug)
        self.printLine()
        print('Density: ', self.rho)
        print('Kinematics Viscosity: ', self.nu)
        print('Grid size: ',self.nx ,' x ', self.ny )        
        print('Domain size: ',self.Lx ,' x ', self.Ly )
        print('Time Step: ', self.dt)
        print('Simulation Time: ', self.tf)
        self.printLine()
        print('Top Wall BC: ', self.u_top)
        print('Bottom Wall BC: ', self.u_bot)
        print('Left Wall BC: ', self.v_left)
        print('Right Wall BC: ', self.v_right)
        self.printLine()
        if self.flagPlot:
            print('Plotting every ', self.plot_frequency, ' steps.')
            self.printLine()  
             
        
                
    def printTimeStatistics(self,val):
        self.flagTimeStatistics = val
    def printDebug(self,str,val):
        # self.printLine('-')
        self.printTextOnLine(str,'-')
        print(val)
    def setKinematicViscosity(self,val):
        self.nu = val    
    def setDensity(self,val):
        self.rho = val
    def setGridPoints(self,nx,ny):
        self.nx = nx
        self.ny = ny
        self.x = np.zeros(shape = nx + 2, dtype = self.__type)
        self.y = np.zeros(shape = ny + 2, dtype = self.__type)
        self.xm = np.zeros(shape = nx + 1, dtype = self.__type)
        self.ym = np.zeros(shape = ny + 1, dtype = self.__type)
    def setDomainSize(self,Lx,Ly):
        self.Lx = Lx
        self.Ly = Ly
    def setTimeStep(self):
        #Convective time step restriction
        CFL = 0.75
        uMax = (max([abs(self.u_top), abs(self.u_bot)])**2.0 
               + max([abs(self.v_left), abs(self.v_right)])**2.0)**0.5
        dt_c = CFL*self.dx/uMax

        #Diffusitve time step restriction
        dt_d = CFL*self.dx**2.0/(4.0*self.nu)

        self.dt = min([dt_c, dt_d])
        if self.debug:
            self.printDebug('Time Step',self.dt)
    def setSimulationTime(self,tf):
        self.tf = tf
        if self.debug:
            self.printDebug('Simulation Time',self.tf)
    def calculateNumberOfTimeSteps(self):
        self.nsteps = int(self.tf/self.dt)
        if self.debug:
            self.printDebug('Total Number of Steps',self.nsteps)
    def createComputationalMesh(self):
        self.imin = 1
        self.imax = self.imin + self.nx - 1
        self.jmin = 1
        self.jmax = self.jmin + self.ny - 1
        self.x[self.imin:self.imax+2] = np.linspace(0,self.Lx,self.nx + 1, dtype=self.__type)
        self.y[self.jmin:self.jmax+2] = np.linspace(0,self.Ly,self.ny + 1, dtype=self.__type)
        self.xm[self.imin:self.imax+1] = 0.5 * (self.x[self.imin:self.imax+1] + self.x[self.imin+1:self.imax+2])
        self.ym[self.jmin:self.jmax+1] = 0.5 * (self.y[self.jmin:self.jmax+1] + self.y[self.jmin+1:self.jmax+2])
        
        self.calculateNumberOfTimeSteps()
        
        # Preallocate Matrices
        self.p=np.zeros((self.imax+2,self.jmax+2), dtype=self.__type)
        self.us=np.zeros((self.imax+2,self.jmax+2), dtype=self.__type)
        self.vs=np.zeros((self.imax+2,self.jmax+2), dtype=self.__type)
        self.L=np.zeros((self.nx*self.ny,self.nx*self.ny), dtype=self.__type)
        self.u=np.zeros((self.imax+2,self.jmax+2), dtype=self.__type)
        self.v=np.zeros((self.imax+2,self.jmax+2), dtype=self.__type)
        
        # Preallocate Arrays
        self.R=np.zeros(self.nx*self.ny, dtype=self.__type)
        self.t=0
        
        self.dx = self.x[self.imin+1] - self.x[self.imin]
        self.dy = self.y[self.jmin+1] - self.y[self.jmin]
        self.dxi = 1/self.dx
        self.dyi = 1/self.dy
    
        
    def setWallVelocity(self,loc,val):
        loc = str.lower(loc)
        if loc == 'top':
            self.u_top = val
        elif loc == 'bottom':
            self.u_bot = val
        elif loc == 'left':
            self.v_left = val
        elif loc == 'right':
            self.v_right = val
        else:
            raise ValueError('Values can only be: top/bottom/left/right')
        self.setTimeStep()
        
        
    def d2_mat_dirichlet_2d(self,nx, ny, dx, dy):
        """
        Constructs the matrix for the centered second-order accurate
        second-order derivative for Dirichlet boundary conditions in 2D

        Parameters
        ----------
        nx : integer
            number of grid points in the x direction
        ny : integer
            number of grid points in the y direction
        dx : float
            grid spacing in the x direction
        dy : float
            grid spacing in the y direction

        Returns
        -------
        d2mat : numpy.ndarray
            matrix to compute the centered second-order accurate first-order deri-
            vative with Dirichlet boundary conditions
        """
        
        nx += 2
        ny += 2
        
        a = 1.0 / dx**2
        g = 1.0 / dy**2
        c = -2.0*a - 2.0*g

        diag_a = a * np.ones((nx-2)*(ny-2)-1)
        diag_a[nx-3::nx-2] = 0.0
        diag_g = g * np.ones((nx-2)*(ny-3))
        diag_c = c * np.ones((nx-2)*(ny-2))

        # We construct a sequence of main diagonal elements,
        diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
        # and a sequence of positions of the diagonal entries relative to the main
        # diagonal.
        offsets = [-(nx-2), -1, 0, 1, nx-2]

        # Call to the diags routine; note that diags return a representation of the
        # array; to explicitly obtain its ndarray realisation, the call to .toarray()
        # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
        d2mat = sp.diags(diagonals, offsets)
        #d2mat = sp.spdiags(diagonals,offsets,nx*nx,ny*ny)
        # print(d2mat.toarray())
        # Return the final array
        return d2mat    
        
        
    def laplacian2D(self,N):
        diag=np.ones([N*N])
        mat=sp.spdiags([diag,-2*diag,diag],[-1,0,1],N,N)
        I=sp.eye(N)
        return np.kron(I,mat)+np.kron(mat,I)
    
    
    def createLaplacian(self):
        for j in range(self.ny):
            for i in range(self.nx):
                self.L[i+(j)*self.nx, i+(j)*self.nx] = 2*self.dxi*self.dxi + 2*self.dyi*self.dyi
                for ii in range(i-1,i+2,2):
                    if ii+1 > 0 and ii+1 <= self.nx:
                        # print(i,j,ii)
                        # print(i+(j)*self.nx,ii+(j)*self.nx)
                        self.L[i+(j)*self.nx,ii+(j)*self.nx] = -self.dxi*self.dxi
                    else:
                        self.L[i+(j)*self.nx,i+(j)*self.nx] += -self.dxi*self.dxi
                for jj in range(j-1,j+2,2):
                    if jj+1 > 0 and jj+1 <= self.ny:
                        # print(i,j,jj)
                        # print(i+(j)*self.nx,i+(jj)*self.nx)
                        self.L[i+(j)*self.nx,i+(jj)*self.nx] = -self.dyi*self.dyi
                    else:
                        self.L[i+(j)*self.nx,i+(j)*self.nx] += -self.dyi*self.dyi
        ind = 0
        self.L[ind,:] = 0
        self.L[ind,ind] = 1

        self.L_sp = sp.csr_matrix(self.L)

        # TO DEBUG - BUILD LAPLACIAN IN SPARSE FORM
        # L2 = self.laplacian2D(self.nx*self.ny)
        # L2 = self.d2_mat_dirichlet_2d(self.nx,self.ny,self.dx,self.dy)    
        # np.set_printoptions(edgeitems=30, linewidth=100000,formatter=dict(float=lambda x: "  %.3g  " % x))
        # print(self.L)
        # print(L2.toarray())
        
        if self.debug:
            np.set_printoptions(edgeitems=30, linewidth=100000,formatter=dict(float=lambda x: "  %.3g  " % x))
            self.printDebug('Laplacian Matrix',self.L)

    def MomentumPredictor(self):
        for j in range(self.jmin,self.jmax+1):
            for i in range(self.imin,self.imax+1):
                A = (self.nu*(self.u[i-1,j]-2*self.u[i,j]+self.u[i+1,j])*self.dxi**2 +
                    self.nu*(self.u[i,j-1]-2*self.u[i,j]+self.u[i,j+1])*self.dyi**2 - 
                    self.u[i,j]*(self.u[i+1,j]-self.u[i-1,j])*0.5*self.dxi -
                    (0.25*(self.v[i-1,j]+self.v[i-1,j+1]+self.v[i,j]+self.v[i,j+1]))*
                    (self.u[i,j+1]-self.u[i,j-1])*0.5*self.dyi)
                B = (self.nu*(self.v[i-1,j]-2*self.v[i,j]+self.v[i+1,j])*self.dxi**2 +
                    self.nu*(self.v[i,j-1]-2*self.v[i,j]+self.v[i,j+1])*self.dyi**2 - 
                    (0.25*(self.u[i,j-1]+self.u[i+1,j-1]+self.u[i,j]+self.u[i+1,j]))*
                    (self.v[i+1,j]-self.v[i-1,j])*0.5*self.dxi -
                    self.v[i,j]*(self.v[i,j+1]-self.v[i,j-1])*0.5*self.dyi)
                self.us[i,j] = self.u[i,j] + self.dt*A
                self.vs[i,j] = self.v[i,j] + self.dt*B
        if self.debug:
            self.printDebug('us',self.us)                

    def computeRHS(self):
        n = 0
        for j in range(self.jmin,self.jmax+1):
            for i in range(self.imin,self.imax+1):
                # print(n, (i-self.imin+1)+(j - self.jmin+1)*(self.jmax-1))
                # print(i,j,n,(j-1)*(self.imax-self.imin + 1) + i - 1)
                self.R[n] = -self.rho/self.dt * (
                               (self.us[i+1,j] - self.us[i,j]) * self.dxi +
                               (self.vs[i,j+1] - self.vs[i,j]) * self.dyi)
                n += 1
        if self.debug:
            self.printDebug('RHS',self.R)   
        
    def calculatePressure(self):
        pv = spl.spsolve(self.L,self.R)
        n = 0
        for j in range(self.jmin,self.jmax+1):
            for i in range(self.imin,self.imax+1):
                self.p[i,j] = pv[n]
                n += 1
        if self.debug:
            self.printDebug('Pressure',self.p)   
                         
    def correctVelocities(self):
        for j in range(self.jmin,self.jmax+1):
            for i in range(self.imin,self.imax+1):
                self.u[i,j] = self.us[i,j] - self.dt/self.rho * (self.p[i,j] - self.p[i-1,j])*self.dxi
                self.v[i,j] = self.vs[i,j] - self.dt/self.rho * (self.p[i,j] - self.p[i,j-1])*self.dyi
        if self.debug:
            self.printDebug('Corrected u (Before BC)',self.u)   
            self.printDebug('Corrected v (Before BC)',self.v)  
        
    def setBoundaryConditions(self,val):
        val = str.lower(val)
        if val == "corrected":
            self.u[:,self.jmin-1] = self.u[:,self.jmin] - 2*(self.u[:,self.jmin] - self.u_bot)
            self.u[:,self.jmax+1] = self.u[:,self.jmax] - 2*(self.u[:,self.jmax] - self.u_top)
            self.v[self.imin-1,:] = self.v[self.imin,:] - 2 *(self.v[self.imin,:] - self.v_left)
            self.v[self.imax+1,:] = self.v[self.imax,:] - 2 *(self.v[self.imax,:] - self.v_right)
            #Set corners to zero
            self.v[0,self.jmax+1] = 0.0
            self.u[self.imax+1,0] = 0.0
            #Set wall velocity to zero
            self.u[1,:] = 0.0
            self.v[:,1] = 0.0
        elif val == "pressure":
            self.p[:,self.jmin-1] = self.p[:,self.jmin]
            self.p[:,self.jmax+1] = self.p[:,self.jmax]
            self.p[self.imin-1,:] = self.p[self.imin,:]
            self.p[self.imax+1,:] = self.p[self.imax,:]
        elif val == "star":
            self.us[:,self.jmin-1] = self.us[:,self.jmin] - 2*(self.us[:,self.jmin] - self.u_bot)
            self.us[:,self.jmax+1] = self.us[:,self.jmax] - 2*(self.us[:,self.jmax] - self.u_top)
            self.vs[self.imin-1,:] = self.vs[self.imin,:] - 2 *(self.vs[self.imin,:] - self.v_left)
            self.vs[self.imax+1,:] = self.vs[self.imax,:] - 2 *(self.vs[self.imax,:] - self.v_right)
            #Set corners to zero
            self.vs[0,self.jmax+1] = 0.0
            self.us[self.imax+1,0] = 0.0
            #Set wall velocity to zero
            self.us[1,:] = 0.0
            self.vs[:,1] = 0.0
        
        if self.debug:
            self.printDebug('Corrected u (After BC)',self.u)   
            self.printDebug('Corrected v (After BC)',self.v) 
            
    def plotContour(self):
        self.checkNaN()
        figure, ax = plt.subplots(figsize=(5,5))
        XX,YY = np.meshgrid(self.x[self.imin:self.imax+1],self.y[self.jmin:self.jmax+1])
        ax.contourf(XX, YY,
                    np.transpose(self.p[self.imin:self.imax+1,self.jmin:self.jmax+1]),
                    10, cmap=plt.cm.bone, origin='lower')
        plt.xlim([0, self.Lx - self.dx])
        plt.ylim([0, self.Ly - self.dy])
        plt.show()       
        
    def plotQuiver(self):
        figure, ax = plt.subplots(figsize=(5,5))
        XX,YY = np.meshgrid(self.x[self.imin:self.imax+1],self.y[self.jmin:self.jmax+1])
        UU = np.transpose(self.u[self.imin:self.imax+1,self.jmin:self.jmax+1])
        VV = np.transpose(self.v[self.imin:self.imax+1,self.jmin:self.jmax+1])
        plt.quiver(XX,YY,UU,VV)
        plt.xlim([0, self.Lx - self.dx])
        plt.ylim([0, self.Ly - self.dy])
        plt.show()       
            
    def initializeFigure(self):
        
        fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize = (16,8))
        fig.suptitle('Results')
        
        XX,YY = np.meshgrid(self.x[self.imin:self.imax+1],self.y[self.jmin:self.jmax+1])
        PP = np.transpose(self.p[self.imin:self.imax+1,self.jmin:self.jmax+1])
        UU = np.transpose(self.u[self.imin:self.imax+1,self.jmin:self.jmax+1]) + 1
        VV = np.transpose(self.v[self.imin:self.imax+1,self.jmin:self.jmax+1])
        
        ax1.contourf(XX,YY,PP)
        ax1.set_xlabel('test')
        ax2.quiver(XX,YY,UU,VV)
        ax2.streamplot(XX,YY,UU,VV,density =2,linewidth=0.5, color='white')

        plt.xlim([0, self.Lx - self.dx])
        plt.ylim([0, self.Ly - self.dy])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)
        return fig, (ax1,ax2)
        
    def updatePlot(self, figure, axes):
        try:
            XX,YY = np.meshgrid(self.x[self.imin:self.imax+1],self.y[self.jmin:self.jmax+1])
            PP = np.transpose(self.p[self.imin:self.imax+1,self.jmin:self.jmax+1])
            UU = np.transpose(self.u[self.imin:self.imax+1,self.jmin:self.jmax+1])
            VV = np.transpose(self.v[self.imin:self.imax+1,self.jmin:self.jmax+1])
            # color = np.sqrt(UU**2 - VV**2)
            
            axes[0].clear()
            axes[1].clear()
            
            ABSUV = np.sqrt(np.add(np.square(UU),np.square(VV)))
            
            axes[0].contourf(XX,YY,PP, levels=30)
            axes[0].set_xlabel('X-Domain')
            axes[0].set_ylabel('Y-Domain')
            axes[0].set_title('Pressure Contour Plot')
            axes[1].contourf(XX,YY,ABSUV, levels=100)
            # axes[1].quiver(XX,YY,UU,VV)
            axes[1].streamplot(XX,YY,UU,VV,density =2,linewidth=0.5,color='white')
            axes[1].set_xlabel('X-Domain')
            axes[1].set_ylabel('Y-Domain')
            axes[1].set_title('Velocity Isolines and Contours')
            plt.xlim([0, self.Lx - self.dx])
            plt.ylim([0, self.Ly - self.dy])
            figure.canvas.draw()
            figure.canvas.flush_events()
            # plt.show(block=False)
        except:
            raise ValueError('ERROR: Model went probably UNSTABLE! Check for NaN!')
            
        
        return figure, axes
        
    @cuda.jit()
    def createLaplacian_kernel(vec,L):
        i,j = cuda.grid(2)
        dims = L.shape
        nx, ny, dxi, dyi = vec
        
        if i >= dims[0] or j >= dims[1]:
            return


        L[i+(j)*nx, i+(j)*nx] = 2*dxi*dxi + 2*dyi*dyi
        for ii in range(i-1,i+2,2):
            if ii +1> 0 and ii +1<= nx:
                L[i+(j)*nx,ii+(j)*nx] = -dxi*dxi
            else:
                L[i+(j)*nx,i+(j)*nx] += -dxi*dxi
        for jj in range(j-1,j+2,2):
            if jj +1> 0 and jj +1<= ny:
                L[i+(j)*nx,i+(jj)*nx] = -dyi*dyi
            else:
                L[i+(j)*nx,i+(j)*nx] += -dyi*dyi
                
        if i == 0 and j == 0:
            L[i,j] = 1
        elif i == 0 and j != 0:
            L[i,j] = 0
        
    @cuda.jit()
    def momentumPredictor_kernel(vec,bds,u,v,us,vs):
        i,j = cuda.grid(2)

        nu, dxi, dyi, dt, rho = vec
        imin, imax, jmin, jmax = bds
        
        if i < imin or i > imax or j < jmin or j > jmax:
            return

        vs[i,j] = v[i,j] + dt*(nu*(v[i-1,j]-2*v[i,j]+v[i+1,j])*dxi**2 +
                                nu*(v[i,j-1]-2*v[i,j]+v[i,j+1])*dyi**2 - 
                                (0.25*(u[i,j-1]+u[i+1,j-1]+u[i,j]+u[i+1,j]))*
                                (v[i+1,j]-v[i-1,j])*0.5*dxi -
                                v[i,j]*(v[i,j+1]-v[i,j-1])*0.5*dyi)
            
        us[i,j] = u[i,j] + dt*(nu*(u[i-1,j]-2*u[i,j]+u[i+1,j])*dxi*dxi +
                                nu*(u[i,j-1]-2*u[i,j]+u[i,j+1])*dyi*dyi - 
                                u[i,j]*(u[i+1,j]-u[i-1,j])*0.5*dxi -
                                (0.25*(v[i-1,j]+v[i-1,j+1]+v[i,j]+v[i,j+1]))*
                                (u[i,j+1]-u[i,j-1])*0.5*dyi)
        
    @cuda.jit()
    def computeRHS_kernel(vec,bds,R,us,vs):
        i,j = cuda.grid(2)
        nu, dxi, dyi, dt, rho = vec
        imin, imax, jmin, jmax = bds
        
        if i < imin or i > imax or j < jmin or j > jmax:
            return
        
        n = (j-1)*imax + (i-1)
        
        R[n] = -rho/dt * ((us[i+1,j] - us[i,j]) * dxi + 
                           (vs[i,j+1] - vs[i,j]) * dyi)
        
    @cuda.jit()
    def calculatePressure_kernel(bds,p,pv):
        i,j = cuda.grid(2)
        imin, imax, jmin, jmax = bds

        if i < imin or i > imax or j < jmin or j > jmax:
            return
        
        n = (j-1)*imax + (i-1)
        
        p[i,j] = pv[n]
        
    @cuda.jit()
    def correct_vel_kernel(vec,bds,u,v,us,vs,p):
        i,j = cuda.grid(2)
        
        nu, dxi, dyi, dt, rho = vec
        imin, imax, jmin, jmax = bds
        
        if i < imin or i > imax or j < jmin or j > jmax:
            return
        
        u[i,j] = us[i,j] - dt/rho * (p[i,j] - p[i-1,j]) * dxi
        v[i,j] = vs[i,j] - dt/rho * (p[i,j] - p[i,j-1]) * dyi 
        
    @cuda.jit()
    def apply_vel_bc_kernel(bds,u,v,vel_bc):
        i,j = cuda.grid(2)
        
        imin, imax, jmin, jmax = bds
        u_bot, u_top, v_left, v_right = vel_bc

        #Tangential BCs
        if i == imin-1:
            v[i,j] = v[i+1,j] - 2 *(v[i+1,j] - v_left)
        if i == imax+1:
            v[i,j] = v[i-1,j] - 2 *(v[i-1,j] - v_right)
        if j == jmin-1:
            u[i,j] = u[i,j+1] - 2 *(u[i,j+1] - u_bot)
        if j == jmax+1:
            u[i,j] = u[i,j-1] - 2 *(u[i,j-1] - u_top)

        #Perpendicular BCs
        if i == imin:
            u[i,j] = 0.0
        if j == jmin:
            v[i,j] = 0.0

        #Corner BCs
        if i == imin-1 and j  == jmax+1:
            v[i,j] = 0.0
        if i == imax+1 and j == jmin-1:
            u[i,j] = 0.0


    @cuda.jit()
    def apply_pres_bc_kernel(bds,p):
        i,j = cuda.grid(2)
        
        imin, imax, jmin, jmax = bds

        if i == imin-1:
            p[i,j] = p[i+1,j]
        if i == imax+1:
            p[i,j] = p[i-1,j]
        if j == jmin-1:
            p[i,j] = p[i,j+1]
        if j == jmax+1:
            p[i,j] = p[i,j-1]
        
    def createLaplacian_parallel(self):
        dims = self.L.shape
        TPB = 16
        vec = np.array([self.nx, self.ny, self.dxi, self.dyi], dtype=np.float32)
        d_vec = cuda.to_device(vec)
        d_L = cuda.to_device(self.L)
        gridDims = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
        blockDims = [TPB, TPB]
        self.createLaplacian_kernel[gridDims,blockDims](d_vec,d_L)

        return d_L.copy_to_host()
    
    def debugGPUmode(self):
        
        self.createComputationalMesh()
        self.createLaplacian()
        
        TPB = 4
        gridDims = [(self.imax+TPB-1)//TPB, (self.jmax+TPB-1)//TPB]
        blockDims = [TPB, TPB]
        
        # Send some quantities over
        bds = np.array([self.imin,self.imax,self.jmin,self.jmax], dtype=np.uint32)
        
        vec = np.array([self.nu, 
                        self.dxi, 
                        self.dyi, 
                        self.dt, 
                        self.rho], 
                        dtype=np.float32)

        vel_bc = np.array([self.u_bot, self.u_top, self.v_left, self.v_right])

        d_vec = cp.asarray(vec)
        d_bds = cp.asarray(bds)
        d_u = cp.asarray(self.u)
        d_v = cp.asarray(self.v)
        d_us = cp.asarray(self.us)
        d_vs = cp.asarray(self.vs)
        d_R = cp.asarray(self.R)
        d_p = cp.asarray(self.p)
        d_pv = cp.zeros_like(d_R)
        d_L = cp.asarray(self.L)
        d_vel_bc = cp.asarray(vel_bc)
        
        self.debug = True
        np.set_printoptions(edgeitems=30, linewidth=100000,formatter=dict(float=lambda x: "  %.3g  " % x))
        
        while self.t <= self.tf:
            self.t += self.dt
            if self.debug:
                print('\n')
                self.printDebug('Time',self.t)
                print('\n')   
                
            # # (1) momentum Predictor
            self.MomentumPredictor()
            self.momentumPredictor_kernel[gridDims,blockDims](d_vec,d_bds,d_u,d_v,d_us,d_vs)
            self.apply_vel_bc_kernel[gridDims,blockDims](d_bds,d_us,d_vs,d_vel_bc)
            us_gpu = cp.asnumpy(d_us)
            self.printDebug('us - GPU',us_gpu) 
            np.testing.assert_allclose(self.us, us_gpu, atol = 1e-3)
            vs_gpu = cp.asnumpy(d_vs)
            self.printDebug('vs - GPU',vs_gpu) 
            np.testing.assert_allclose(self.vs, vs_gpu, atol = 1e-3)
            
            # (2) RHS
            self.computeRHS()
            self.computeRHS_kernel[gridDims,blockDims](d_vec,d_bds,d_R,d_us,d_vs)
            # RHS_gpu = d_R.copy_to_host()
            RHS_gpu = cp.asnumpy(d_R)
            self.printDebug('RHS - GPU',RHS_gpu)
            np.testing.assert_allclose(self.R, RHS_gpu, atol = 1e-3)
            
            # (2bis) Poisson Step
            d_pv = cp.linalg.solve(d_L,d_R)
            
            # (3) Calculate Pressure
            self.calculatePressure()
            self.calculatePressure_kernel[gridDims,blockDims](d_bds,d_p,d_pv)
            self.apply_pres_bc_kernel[gridDims,blockDims](d_bds,d_p)
            # p_gpu = d_p.copy_to_host()
            p_gpu = cp.asnumpy(d_p)
            self.printDebug('Pressure - GPU',p_gpu)
            np.testing.assert_allclose(self.p, p_gpu, atol = 1e-3)
            
            # (4) u,v corrected
            self.debug = False
            self.correctVelocities()
            self.debug = True
            self.correct_vel_kernel[gridDims,blockDims](d_vec,d_bds,d_u,d_v,d_us,d_vs,d_p)
            u_gpu = cp.asnumpy(d_u)
            v_gpu = cp.asnumpy(d_v)
            self.printDebug('Corrected u (Before BC)',self.u)  
            self.printDebug('Corrected u (Before BC) - GPU',u_gpu)
            self.printDebug('Corrected v (Before BC)',self.v)  
            self.printDebug('Corrected v (Before BC) - GPU',v_gpu)
            np.testing.assert_allclose(self.u, u_gpu, atol = 1e-3)
            np.testing.assert_allclose(self.v, v_gpu, atol = 1e-3)
            
            # (5) Apply BC
            self.setBoundaryConditions('corrected')
            self.apply_vel_bc_kernel[gridDims,blockDims](d_bds,d_u,d_v,d_vel_bc)
            # u_gpu = d_u.copy_to_host()
            # v_gpu = d_v.copy_to_host()
            u_gpu = cp.asnumpy(d_u)
            v_gpu = cp.asnumpy(d_v)
            self.printDebug('Corrected u (After BC) - GPU',u_gpu)
            self.printDebug('Corrected v (After BC) - GPU',v_gpu)
            np.testing.assert_allclose(self.u, u_gpu, atol = 1e-3)
            np.testing.assert_allclose(self.v, v_gpu, atol = 1e-3)
            
    def checkNaN(self):
        pass
            
    def solve_parallel(self):
        # cuda.profile_start()
        TPB = 4
        
        bds = np.array([self.imin,self.imax,self.jmin,self.jmax], dtype=np.uint32)
        
        vec = np.array([self.nu, 
                        self.dxi, 
                        self.dyi, 
                        self.dt, 
                        self.rho], 
                        dtype=self.__type)

        vel_bc = np.array([self.u_bot, self.u_top, self.v_left, self.v_right])
        
        d_vec = cp.asarray(vec)
        d_bds = cp.asarray(bds)
        d_u = cp.asarray(self.u)
        d_v = cp.asarray(self.v)
        d_us = cp.asarray(self.us)
        d_vs = cp.asarray(self.vs)
        d_R = cp.asarray(self.R)
        d_p = cp.asarray(self.p)
        d_pv = cp.zeros_like(d_R)
        d_L = cp.asarray(self.L)
        d_vel_bc = cp.asarray(vel_bc)
        
        d_L_sp = cupyx.scipy.sparse.csr_matrix(self.L_sp)
        
        gridDims = [(self.imax+TPB-1)//TPB, (self.jmax+TPB-1)//TPB]
        blockDims = [TPB, TPB]
        
        # stream1 = cuda.stream()
        # stream2 = cuda.stream()
        bar = Bar('Computing',max=self.nsteps)
        it = 1
        first_plot = True
        start = time.time()
        while self.t <= self.tf:
            self.t += self.dt
            self.momentumPredictor_kernel[gridDims,blockDims](d_vec,d_bds,d_u,d_v,d_us,d_vs)
            self.apply_vel_bc_kernel[gridDims,blockDims](d_bds,d_us,d_vs,d_vel_bc)
            self.computeRHS_kernel[gridDims,blockDims](d_vec,d_bds,d_R,d_us,d_vs)
            d_pv = cpl.spsolve(d_L_sp,d_R)
            #d_pv = cp.linalg.solve(d_L,d_R)
            self.calculatePressure_kernel[gridDims,blockDims](d_bds,d_p,d_pv)
            self.apply_pres_bc_kernel[gridDims,blockDims](d_bds,d_p)
            self.correct_vel_kernel[gridDims,blockDims](d_vec,d_bds,d_u,d_v,d_us,d_vs,d_p)
            self.apply_vel_bc_kernel[gridDims,blockDims](d_bds,d_u,d_v,d_vel_bc)
            if first_plot and self.flagPlot:
                fig, axes = self.initializeFigure()
                first_plot = False
            if it % self.plot_frequency == 0 and self.flagPlot: 
                self.p = cp.asnumpy(d_p)
                self.u = cp.asnumpy(d_u)
                self.v = cp.asnumpy(d_v)
                fig, axes = self.updatePlot(fig, axes)
            bar.next()
            it += 1
        # cuda.profile_stop()
        bar.finish()
        end = time.time()
        self.p = cp.asnumpy(d_p)
        self.u = cp.asnumpy(d_u)
        self.v = cp.asnumpy(d_v)
        return float(end-start)

        
    def solve(self):
        
        self.createComputationalMesh()
        self.setBoundaryConditions('corrected')
        self.createLaplacian()
        
        if self.verbose:
            self.printHeaderAndOptions()
        
        
        if self.flagGPU:
            time_elapsed = self.solve_parallel()
        else:
            first_plot = True
            it = 1
            bar = Bar('Computing',max=self.nsteps)
            start = time.time()
            while self.t <= self.tf:
                # Update Time
                self.t += self.dt
                if self.debug:
                    print('\n')
                    self.printDebug('Time',self.t)
                    print('\n')
                self.MomentumPredictor()
                self.setBoundaryConditions('star')
                self.computeRHS()
                self.calculatePressure()
                self.setBoundaryConditions('pressure')
                self.correctVelocities()
                self.setBoundaryConditions('corrected')
                if first_plot and self.flagPlot:
                    fig, axes = self.initializeFigure()
                    first_plot = False
                if it % self.plot_frequency == 0 and self.flagPlot: 
                    fig, axes = self.updatePlot(fig, axes)
                bar.next()
                it += 1
            bar.finish()
            time_elapsed = time.time() - start
            self.updatePlot(fig,axes)
            
        if self.flagTimeStatistics:
            self.printTextOnLine('Time Statistics','-')
            if self.flagGPU:
                print('GPU Mode: ', end = '')    
            else:
                print('CPU Mode: ', end = '')  
            print(' ' + str(round(time_elapsed, self.sig_figs)) + ' seconds!')
            
        
        plt.show()
        
    def runBenchmark(self, N = None):
        
        # Run N iteration CPU vs GPU
        if N is None:
            N = self.nsteps
            
        self.createComputationalMesh()
        self.setBoundaryConditions('corrected')
        self.createLaplacian()

        self.debug = False
        self.flagPlot = False
        self.flagGPU = True
        
        self.printHeaderAndOptions() 
        self.printTextOnLine(('CPU/GPU Benchmark for ' + str(N) + ' iterations'))  
        self.printLine()
            
        # Run CPU First
        bar = Bar('CPU Run ',max=N)
        it = 1
        start = time.time()
        while it <= N:
            # Update Time
            self.t += self.dt
            self.MomentumPredictor()
            self.setBoundaryConditions('star')
            self.computeRHS()
            self.calculatePressure()
            self.setBoundaryConditions('pressure')
            self.correctVelocities()
            self.setBoundaryConditions('corrected')
            bar.next()
            it += 1
        end = time.time()
        bar.finish()
        cpu_time = float(end-start)
        us_cpu = self.us
        vs_cpu = self.vs
        R_cpu = self.R
        p_cpu = self.p
        u_cpu = self.u
        v_cpu = self.v
        
        # Reset Values
        self.createComputationalMesh()
        L_old = self.L
        self.createLaplacian()
        L_new = self.L
        # Just checking 
        np.testing.assert_allclose(L_old, L_new, atol = 1e-3)
        L_old = None
        L_new = None
        
        # Run GPU
        TPB = 4
        
        bds = np.array([self.imin,self.imax,self.jmin,self.jmax], dtype=np.uint32)
        
        vec = np.array([self.nu, 
                        self.dxi, 
                        self.dyi, 
                        self.dt, 
                        self.rho], 
                        dtype=self.__type)

        vel_bc = np.array([self.u_bot, self.u_top, self.v_left, self.v_right])
        
        d_vec = cp.asarray(vec)
        d_bds = cp.asarray(bds)
        d_u = cp.asarray(self.u)
        d_v = cp.asarray(self.v)
        d_us = cp.asarray(self.us)
        d_vs = cp.asarray(self.vs)
        d_R = cp.asarray(self.R)
        d_p = cp.asarray(self.p)
        d_pv = cp.zeros_like(d_R)
        d_L = cp.asarray(self.L)
        d_vel_bc = cp.asarray(vel_bc)

        d_L_sp = cupyx.scipy.sparse.csr_matrix(self.L_sp)

        gridDims = [(self.imax+TPB-1)//TPB, (self.jmax+TPB-1)//TPB]
        blockDims = [TPB, TPB]

        self.apply_vel_bc_kernel[gridDims,blockDims](d_bds,d_u,d_v,d_vel_bc)
        
        bar = Bar('GPU Run ',max=N)
        it = 1
        start = time.time()
        while it <= N:
            self.t += self.dt
            self.momentumPredictor_kernel[gridDims,blockDims](d_vec,d_bds,d_u,d_v,d_us,d_vs)
            self.apply_vel_bc_kernel[gridDims,blockDims](d_bds,d_us,d_vs,d_vel_bc)
            self.computeRHS_kernel[gridDims,blockDims](d_vec,d_bds,d_R,d_us,d_vs)
            d_pv = cpl.spsolve(d_L_sp,d_R)
            self.calculatePressure_kernel[gridDims,blockDims](d_bds,d_p,d_pv)
            self.apply_pres_bc_kernel[gridDims,blockDims](d_bds,d_p)
            self.correct_vel_kernel[gridDims,blockDims](d_vec,d_bds,d_u,d_v,d_us,d_vs,d_p)
            self.apply_vel_bc_kernel[gridDims,blockDims](d_bds,d_u,d_v,d_vel_bc)
            bar.next()
            it += 1
        end = time.time()
        bar.finish()
        gpu_time = float(end-start)
        us_gpu = cp.asnumpy(d_us)
        vs_gpu = cp.asnumpy(d_vs)
        R_gpu = cp.asnumpy(d_R)
        p_gpu = cp.asnumpy(d_p)
        u_gpu = cp.asnumpy(d_u)
        v_gpu = cp.asnumpy(d_v)
        
        self.printTextOnLine('Time Statistics','-')
        print('CPU Time: ', round(cpu_time,self.sig_figs),'s')
        print('GPU Time: ', round(gpu_time,self.sig_figs),'s')
        print('Speed-up Factor: ' +  str(round(cpu_time/gpu_time,self.sig_figs)) + 'x')
        
        self.printTextOnLine('Check Accuracy','-')
        print('Pressure Norm-L2 Value: ', round(np.linalg.norm(p_cpu - p_gpu), self.sig_figs))
        print('u-vel Norm-L2 Value: ', round(np.linalg.norm(u_cpu - u_gpu), self.sig_figs)) 
        print('v-vel Norm-L2 Value: ', round(np.linalg.norm(v_cpu - v_gpu), self.sig_figs))
        
        np.testing.assert_allclose(us_cpu[self.imin:self.imax,self.jmin:self.jmax], 
                                        us_gpu[self.imin:self.imax,self.jmin:self.jmax], atol = 1e-1, err_msg='us')
        np.testing.assert_allclose(vs_cpu[self.imin:self.imax,self.jmin:self.jmax], 
                                        vs_gpu[self.imin:self.imax,self.jmin:self.jmax], atol = 1e-1, err_msg='vs')
        np.testing.assert_allclose(R_cpu, R_gpu, atol = 1e-1, err_msg='R')
        np.testing.assert_allclose(p_cpu[self.imin:self.imax,self.jmin:self.jmax], 
                                        p_gpu[self.imin:self.imax,self.jmin:self.jmax], atol = 1e-1, err_msg='p')
        np.testing.assert_allclose(u_cpu[self.imin:self.imax,self.jmin:self.jmax], 
                                        u_gpu[self.imin:self.imax,self.jmin:self.jmax], atol = 1e-1, err_msg='u')
        np.testing.assert_allclose(v_cpu[self.imin:self.imax,self.jmin:self.jmax], 
                                        v_gpu[self.imin:self.imax,self.jmin:self.jmax], atol = 1e-1, err_msg='v')
        
        

def main():
    clearConsole()
    test = TPSolver(False)
    test.enableGPU(True)
    test.setVerbose(True)
    test.setDebug(False)
    test.setDensity(1.225)
    test.setKinematicViscosity(0.005)
    test.setGridPoints(50,50)
    test.setDomainSize(1,1)
    test.setSimulationTime(20)
    test.printTimeStatistics(True)   
    test.createComputationalMesh()
    test.setWallVelocity('top',4)
    test.setWallVelocity('right',-4)
    test.setWallVelocity('bottom',-4)
    test.setWallVelocity('left',4)
    #test.setTimeStep() #This is called within the setWallVelocity method
    test.plotEveryNTimeSteps(10)
    
    test.solve()
    # test.debugGPUmode()
    # test.runBenchmark(100)


if __name__ == '__main__':
    main()  
