import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


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
    
    # INDICES
    imin = 0
    imax = 0
    jmin = 0
    jmax = 0
    
    __type = np.float32

    # Matrices
    x = []
    y = []
    xm = []
    ym = []
    p = []
    us = []
    vs = []
    R = []
    u = []
    v = []
    t = []
    L = []
    
    # Initial Velocities
    u_bot = 0
    u_top = 0
    v_left = 0
    v_right = 0
    
    # Mesh size
    dx = 0
    dy = 0
    dxi = 0
    dxy = 0
    
    debug = False
    
    def __init__(self):
        pass
    
    def setDebug(self,val):
        self.debug = val
    def printDebug(self,str,val):
        print('---------------------------------------')
        print(str)
        print(val)
    def setKinematicViscosity(self,val):
        self.nu = val    
    def setDensity(self,val):
        self.rho = val
    def setGridPoints(self,nx,ny):
        self.nx = nx
        self.ny = ny
        self.x = np.zeros(shape = nx + 2, dtype = self.__type)
        self.y = np.zeros(shape = nx + 2, dtype = self.__type)
        self.xm = np.zeros(shape = nx+ 1, dtype = self.__type)
        self.ym = np.zeros(shape = ny+ 1, dtype = self.__type)
    def setDomainSize(self,Lx,Ly):
        self.Lx = Lx
        self.Ly = Ly
    def setTimeStep(self,dt):
        self.dt = dt
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
        self.imin = 2
        self.imax = self.imin + self.nx - 1
        self.jmin = 2
        self.jmax = self.jmin + self.ny - 1
        self.x[self.imin-1:self.imax+1] = np.linspace(0,self.Lx,self.nx + 1, dtype=self.__type)
        self.y[self.jmin-1:self.jmax+1] = np.linspace(0,self.Lx,self.nx + 1, dtype=self.__type)
        self.xm[self.imin-1:self.imax] = 0.5 * (self.x[self.imin-1:self.imax] + self.x[self.imin:self.imax+1])
        self.ym[self.jmin-1:self.jmax] = 0.5 * (self.y[self.jmin-1:self.jmax] + self.y[self.jmin:self.jmax+1])
        
        self.calculateNumberOfTimeSteps()
        
        # Preallocate Matrices
        self.p=np.zeros((self.imax,self.jmax), dtype=self.__type)
        self.us=np.zeros((self.imax+1,self.jmax+1), dtype=self.__type)
        self.vs=np.zeros((self.imax+1,self.jmax+1), dtype=self.__type)
        self.L=np.zeros((self.nx*self.ny,self.nx*self.ny), dtype=self.__type)
        self.u=np.zeros((self.imax+1,self.jmax+1), dtype=self.__type)
        self.v=np.zeros((self.imax+1,self.jmax+1), dtype=self.__type)
        
        # Preallocate Arrays
        self.R=np.zeros(self.nx*self.ny, dtype=self.__type)
        self.t=0
        # self.Z=np.peaks(nx);
        
        self.dx = self.x[self.imin] - self.x[self.imin-1]
        self.dy = self.y[self.jmin] - self.y[self.jmin-1]
        self.dxi = 1/self.dx
        self.dyi = 1/self.dy
        
    def setInitialVelocity(self,loc,val):
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
  
    def createLaplacian(self):
        for j in range(self.ny):
            for i in range(self.nx):
                self.L[i+(j)*self.nx, i+(j)*self.nx] = 2*self.dxi*self.dxi + 2*self.dyi*self.dyi
                for ii in range(i-1,i+2,2):
                    if ii +1> 0 and ii +1<= self.nx:
                        # print(i,j,ii)
                        # print(i+(j)*self.nx,ii+(j)*self.nx)
                        self.L[i+(j)*self.nx,ii+(j)*self.nx] = -self.dxi*self.dxi
                    else:
                        self.L[i+(j)*self.nx,i+(j)*self.nx] += -self.dxi*self.dxi
                for jj in range(j-1,j+2,2):
                    if jj +1> 0 and jj +1<= self.ny:
                        # print(i,j,jj)
                        # print(i+(j)*self.nx,i+(jj)*self.nx)
                        self.L[i+(j)*self.nx,i+(jj)*self.nx] = -self.dyi*self.dyi
                    else:
                        self.L[i+(j)*self.nx,i+(j)*self.nx] += -self.dyi*self.dyi
        self.L[0,:] = 0
        self.L[0,0] = 1
        if self.debug:
            np.set_printoptions(edgeitems=30, linewidth=100000,formatter=dict(float=lambda x: "  %.3g  " % x))
            self.printDebug('Laplacian Matrix',self.L)

    def uMomentumPredictor(self):
        for j in range(self.jmin-1,self.jmax):
            for i in range(self.imin,self.imax):
                A = (self.nu*(self.u[i-1,j]-2*self.u[i,j]+self.u[i+1,j])*self.dxi**2 +
                    self.nu*(self.u[i,j-1]-2*self.u[i,j]+self.u[i,j+1])*self.dyi**2 - 
                    self.u[i,j]*(self.u[i+1,j]-self.u[i-1,j])*0.5*self.dxi -
                    (0.25*(self.v[i-1,j]+self.v[i-1,j+1]+self.v[i,j]+self.v[i,j+1]))*
                    (self.u[i,j+1]-self.u[i,j-1])*0.5*self.dyi)
                # print(i,j)
                self.us[i,j] = self.u[i,j] + self.dt*A
        if self.debug:
            self.printDebug('us',self.us)                
                
    def vMomentumPredictor(self):
        for j in range(self.jmin,self.jmax):
            for i in range(self.imin-1,self.imax):
                B = (self.nu*(self.v[i-1,j]-2*self.v[i,j]+self.v[i+1,j])*self.dxi**2 +
                    self.nu*(self.v[i,j-1]-2*self.v[i,j]+self.v[i,j+1])*self.dyi**2 - 
                    (0.25*(self.u[i,j-1]+self.u[i+1,j-1]+self.u[i,j]+self.u[i+1,j]))*
                    (self.v[i+1,j]-self.v[i-1,j])*0.5*self.dxi -
                    self.v[i,j]*(self.v[i,j+1]-self.v[i,j-1])*0.5*self.dyi)
                # print(i,j)
                self.vs[i,j] = self.v[i,j] + self.dt*B
        if self.debug:
            self.printDebug('vs',self.vs)        
                
    def computeRHS(self):
        n = 0
        for j in range(self.jmin-1,self.jmax):
            for i in range(self.imin-1,self.imax):
                # print(i,j)
                n += 1
                self.R[n-1] = -self.rho/self.dt * (
                               (self.us[i+1,j] - self.us[i,j]) * self.dxi +
                               (self.vs[i,j+1] - self.vs[i,j]) * self.dyi)
        if self.debug:
            self.printDebug('RHS',self.R)   
        
    def calculatePressure(self):
        pv = np.linalg.solve(self.L,self.R)
        n = 0
        for j in range(self.jmin-1,self.jmax):
            for i in range(self.imin-1,self.imax):
                self.p[i,j] = pv[n]
                n += 1
        if self.debug:
            self.printDebug('Pressure',self.p)   
                         
    def correctVelocities(self):
        for j in range(self.jmin-1,self.jmax):
            for i in range(self.imin,self.imax):
                self.u[i,j] = self.us[i,j] - self.dt/self.rho * (self.p[i,j] - self.p[i-1,j])*self.dxi
        for j in range(self.jmin,self.jmax):
            for i in range(self.imin-1,self.imax):
                self.v[i,j] = self.vs[i,j] - self.dt/self.rho * (self.p[i,j] - self.p[i,j-1])*self.dyi
        if self.debug:
            self.printDebug('Corrected u (Before BC)',self.u)   
            self.printDebug('Corrected v (Before BC)',self.v)  
        
    def setBoundaryConditions(self):
        self.u[:,self.jmin-2] = self.u[:,self.jmin-1] - 2*(self.u[:,self.jmin-1] - self.u_bot)
        self.u[:,self.jmax] = self.u[:,self.jmax-1] - 2*(self.u[:,self.jmax-1] - self.u_top)
        self.v[self.imin-2,:] = self.v[self.imin-1,:] - 2 *(self.v[self.imin-1,:] - self.v_left)
        self.v[self.imax,:] = self.v[self.imax-1,:] - 2 *(self.v[self.imax-1,:] - self.v_right)
        if self.debug:
            self.printDebug('Corrected u (After BC)',self.u)   
            self.printDebug('Corrected v (After BC)',self.v) 
            
    def initializeFigure(self):
        figure, ax = plt.subplots(figsize=(5,5))
        XX,YY = np.meshgrid(self.x[self.imin-1:self.imax],self.y[self.jmin-1:self.jmax])
        CS = ax.contourf(XX, YY,
                    np.transpose(self.p[self.imin-1:self.imax,self.jmin-1:self.jmax]),
                    10, cmap=plt.cm.bone, origin='lower')
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")

        cbar = plt.colorbar(CS, cax=cax)
        cbar.ax.set_ylabel('Pressure Pascal [P]')
        plt.xlim([0, self.Lx - self.dx])
        plt.ylim([0, self.Ly - self.dy])
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.show()
        return figure, ax, cax
        
    def updatePlot(self, figure, ax, cax):
        XX,YY = np.meshgrid(self.x[self.imin-1:self.imax],self.y[self.jmin-1:self.jmax])
        CS = ax.contourf(XX, YY,
                    np.transpose(self.p[self.imin-1:self.imax,self.jmin-1:self.jmax]),
                    10, cmap=plt.cm.bone, origin='lower')
        cbar = plt.colorbar(CS, cax=cax)
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.show()
        
        
    def solve(self):
        
        first_plot = True
        while self.t <= self.tf:
            # Update Time
            self.t += self.dt
            if self.debug:
                print('\n')
                self.printDebug('Time',self.t)
                print('\n')
            self.uMomentumPredictor()
            self.vMomentumPredictor()
            self.computeRHS()
            self.calculatePressure()
            self.correctVelocities()
            self.setBoundaryConditions()
            if first_plot:
                fig, ax, cax = self.initializeFigure()
                first_plot = False
            self.updatePlot(fig, ax, cax)





                
            

def main():
    test = TPSolver()
    test.setDebug(False)
    test.setDensity(1.225)
    test.setKinematicViscosity(0.001)
    test.setGridPoints(40,40)
    test.setDomainSize(1,1)
    test.setTimeStep(0.001)
    test.setSimulationTime(1)
    
    test.createComputationalMesh()
    test.setInitialVelocity('top',2)
    test.setInitialVelocity('bottom',2)
    test.setInitialVelocity('right',1)
    test.createLaplacian()
    
    test.solve()


if __name__ == '__main__':
    main()  
