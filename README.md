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

## Theory
### Governing Equations
```math
e^{i\pi} + 1 = 0
```
