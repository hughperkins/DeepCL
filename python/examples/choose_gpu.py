"""
Example of choosing gpu from python
"""
import PyDeepCL

# gpu 0
cl0 = PyDeepCL.DeepCL()

# gpu 1
cl1 = PyDeepCL.DeepCL(gpuindex=1)

