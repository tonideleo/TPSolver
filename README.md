# TPSolver (The Toni and Pablo Solver)

This is a python class used to discretize and run a 2D Navier-Stokes fluid solver.
It is meant to run on both CPU and GPU (using Cuda, Numba, Numpy, and Cupy) to show how beneficial the GPU parallel implementation can be.

## Prerequesites
The following modules are called at the beginning of the code:

```py 
import numpy as np                # pip install numpy
import cupy as cp                 # pip install cupy
import scipy.sparse as sp         # pip install scipy
import matplotlib.pyplot as plt   # pip install matplotlib
import time
from numba import cuda            # pip install numba
from progress.bar import Bar      # pip install progress
from cpuinfo import get_cpu_info  # pip install py-cpuinfo
import os
```

The GPU implementation requires a working cuda device (AMD devices have not been tested). More information can be found at:
 - [Numba Website](https://numba.pydata.org/)
 - [Cupy Website](https://cupy.dev/)

## Usage
```py
test = TPSolver()
test.enableGPU(True)
test.setVerbose(True)
test.setDebug(False)
test.setDensity(1.225)
test.setKinematicViscosity(0.005)
test.setGridPoints(30,30)
test.setDomainSize(1,1)
test.setTimeStep(0.001)
test.setSimulationTime(20)
test.printTimeStatistics(True)   
test.setInitialVelocity('top',4)
test.setInitialVelocity('right',4)
test.setInitialVelocity('bottom',-4)
test.setInitialVelocity('left',-4)
test.plotEveryNTimeSteps(30)

test.solve()
# test.debugGPUmode()
# test.runBenchmark(300)
```

