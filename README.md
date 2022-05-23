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
An instance of the TPSolver class can be instantiated with the following snippet:
```py
TP = TPSolver(license = True, GPUmode = False)
```
where `license = true` is the default value to print out the license warning and `GPUmode = false` is the default value for running on the CPU.
At this point, any interaction with the class can be done with the following methods:

### Initialization Methods:
#### Enable GPU:
If this flag is activated, in normal run, the solver will run using the GPU.
```py
TP.enableGPU(True)
```

#### Set Verbose Printout:
If this flag is activated, the solver will print out verbose options.
```py
TP.setVerbose(True)
```

```py
TP.setDebug(False)
TP.setDensity(1.225)
TP.setKinematicViscosity(0.005)
TP.setGridPoints(30,30)
TP.setDomainSize(1,1)
TP.setTimeStep(0.001)
TP.setSimulationTime(20)
TP.printTimeStatistics(True)   
TP.setInitialVelocity('top',4)
TP.setInitialVelocity('right',4)
TP.setInitialVelocity('bottom',-4)
TP.setInitialVelocity('left',-4)
TP.plotEveryNTimeSteps(30)

test.solve()
# test.debugGPUmode()
# test.runBenchmark(300)
```

