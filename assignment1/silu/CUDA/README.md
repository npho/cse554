# SiLU CUDA Kernel

### Compile and Test

```bash
$ mkdir build
$ cmake -B build
-- The CXX compiler identification is GNU 8.5.0
-- The CUDA compiler identification is NVIDIA 12.9.86 with host compiler GNU 8.5.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /sw/cuda/12.9.1/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Configuring done (1.6s)
-- Generating done (0.0s)
-- Build files have been written to: /gscratch/scrubbed/npho/cse554/assignment1/silu/CUDA/build
$ cd build
$ make
$ ./silu
================================================================================
SiLU Kernel Tests
================================================================================
CUDA device:    Quadro RTX 6000
CPU time:       318.0902 ms
GPU time:       0.9754 ms
Speedup:        326.12x
Bandwidth:      550.42 GB/s
Max error:      1.91e-06
Status:         PASSED

$
```

### NCU Profiling

```bash
$ ncu -o cuda_silu.ncu-rep -f ./silu
```
