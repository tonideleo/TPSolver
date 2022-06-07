# TPSolver (The Toni and Pablo Solver)

This is a python class used to discretize and run a 2D Navier-Stokes fluid solver.
It is meant to run on both CPU and GPU (using Cuda, Numba, Numpy, and Cupy) to show how beneficial the GPU parallel implementation can be.

## Prerequesites
The following modules are called at the beginning of the code:

```py 
# Required imports (some might require installation)
import numpy as np                # pip install numpy
import cupy as cp                 # pip install cupy
import scipy.sparse as sp         # pip install scipy
import matplotlib.pyplot as plt   # pip install matplotlib
from numba import cuda            # pip install numba
from progress.bar import Bar      # pip install progress
from cpuinfo import get_cpu_info  # pip install py-cpuinfo
# Derived imports
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg as cpl
# System imports
import os
import warnings
import time
```

The GPU implementation requires a working cuda device (AMD devices have not been tested). More information can be found at:
 - [Numba Website](https://numba.pydata.org/)
 - [Cupy Website](https://cupy.dev/)

You should be careful in matching the version of CUDA installed with the version of cupy.
Install order might also be important, it is recommended to create a new conda environment:
```py
conda create --name cuda
conda activate cuda
```
and install the following packages in order:
```py
pip install numpy, scipy, matplotlib, numba, cupy, progress, py-cpuinfo
```
Using `pip install cupy` is recommened as the package will be compiled from source and be the most up-to-date.

The only disadvantage is that it might take a long time (up to 20 minutes on some machines).

If using wheel packages such as `pip install cupy-cuda116` you must be sure to have CUDA 11.6 installed. A mismatch in version won't allow to import cupy.

These are pre-built binaries and will be installed in seconds; however, they might not be the most up-to-date.

For Windows, for example, version 8.x.x will be installed.

The sparse implementation of the Laplacian WILL NOT work if cupy version is less than 10.x.x.

## Quick Overview
The following methods are available to use:

```py
TP = TPSolver(False)                    # False flag is to suppress license printout
TP.enableGPU(True)                      # Enable GPU mode
TP.enableSparseL(False)                 # Enable Sparse Matrices
TP.setFloatType(32)                     # Specify Precision (32/64)
#
TP.setTPBX(x)                           # Set Threads per block in X-dir
TP.setTPBY(y)                           # Set Threads per block in Y-dir
TP.setTPB(x,y)                          # Set Threads per block in both X and Y dir
#
TP.setCFL(0.75)                         # Set Courant–Friedrichs–Lewy condition
#
TP.setVerbose(True)                     # Set Verbose output
TP.setDebug(False)                      # Set Debug ouput (print of ALL matrices)
#
TP.setDensity(1.225)                    # Set Fluid Density
TP.setKinematicViscosity(0.005)         # Set Kinematics Viscosity of the Fluid
#
TP.setGridPoints(50,50)                 # Set Grid Dimensions (X and Y dir)
TP.setDomainSize(1, 1)                  # Set Domain Dimensions (X and Y dir)
TP.setSimulationTime(20)                # Set Total Time Simulation
# 
TP.printTimeStatistics(True)            # Print Time Statistics at the end of the run
TP.setWallVelocity('top', 4)            # Set Wall Velocities (top,bottom,left,right)
#  
TP.plotEveryNTimeSteps(10)              # Plot every N time steps
TP.savePlots(True)                      # Save pdf plot at the end of the run

# Principal Calling Methods (run one only at the time!)
TP.solve()                              # Run the problem until completion
TP.debugGPUmode()                       # Run both CPU and GPU and compare each matrix at each time steps
TP.runBenchmark(Niter)                  # Run both CPU and GPU for Niter iterations and compare at the end

# Sweep different grid dimensions (square domain only) and compare CPU vs GPU
# Prepare plot at the end: grid dimensions vs speed-up factor
# Run Niter iterations and grid sweep dimensions are given by:
# min = Minimum Grid Size Dimensions
# max = Maximum Grid Size Dimensions
# steps = integer number of steps between min and max
TP.sweepGridDimensionsBenchmark(Niter,min,max,steps) 
```

## Detailed Overview
An instance of the TPSolver class can be instantiated with the following snippet:
```py
TP = TPSolver(license = True)
```
where `license = true` is the default value to print out the license warning.
At this point, any interaction with the class can be done with the following methods:

### Enable GPU
```py
TP.enableGPU(True)                      # Enable GPU mode
```
If this flag is activated, in normal run `TP.solve()`, the solver will run using the GPU.

### Enable Verbose Printout
If this flag is activated, the solver will print out verbose options.
```py
TP.setVerbose(True)                     # Set Verbose output
```

### Enable Sparse Laplacian Implementation
```py
TP.enableSparseL(False)                 # Enable Sparse Matrices
```
With this flag, the Laplacian will be constructed in sparse form.
This will allow much larger problem to be run and massive speed-up on the cpus.
However, on the GPU the CUSparse implementation is not optimized and unfortunately will run much slower on the GPU as of today.

### Set Precision Type
```py
TP.setFloatType(32)                     # Specify Precision (32/64)
```
The only two options are 32 (for np.float32) or 64 (for np.float64). 
Half precision (np.float16) is not supported and long double (np.float128) is not implemented (as it is not supported on most GPUs).

