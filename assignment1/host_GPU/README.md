# Host-Device Memory Transfer Benchmarking

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
-- Configuring done (5.4s)
-- Generating done (0.0s)
-- Build files have been written to: /gscratch/scrubbed/npho/cse554/assignment1/host_GPU/build
$ cd build
$ make
$ ./copy
CUDA Device: Quadro RTX 6000

   Bytes    H2D Page    D2H Page  H2D Pinned  D2H Pinned H2D Speedup D2H Speedup
--------------------------------------------------------------------------------
       1       0.000       0.000       0.000       0.000       0.880       1.110
       2       0.000       0.000       0.000       0.000       0.887       1.115
       4       0.001       0.001       0.001       0.001       0.903       1.105
       8       0.001       0.001       0.001       0.001       0.891       1.085
      16       0.003       0.002       0.002       0.002       0.889       1.164
      32       0.005       0.004       0.005       0.005       0.902       1.153
      64       0.011       0.008       0.010       0.010       0.889       1.129
     128       0.021       0.017       0.019       0.019       0.900       1.135
     256       0.043       0.033       0.038       0.038       0.887       1.144
     512       0.086       0.065       0.077       0.075       0.897       1.152
    1024       0.166       0.131       0.148       0.149       0.893       1.140
    2048       0.318       0.254       0.291       0.295       0.914       1.160
    4096       0.607       0.473       0.522       0.585       0.859       1.238
    8192       0.959       0.804       0.897       1.111       0.935       1.383
   16384       1.454       1.315       1.529       2.059       1.051       1.565
   32768       2.192       1.926       3.115       3.524       1.421       1.830
   65536       2.621       2.501       4.810       5.466       1.835       2.185
  131072       3.152       2.946       6.942       7.795       2.203       2.646
  262144       3.493       3.233       8.875       9.779       2.541       3.025
  524288       3.658       3.366      10.229      11.152       2.796       3.313
 1048576       3.748       3.405      11.021      12.030       2.941       3.533

Results saved to: benchmark.csv

================================================================================
Strided Column Copy Benchmark (copy_first_column)
================================================================================
      Matrix: 8192 x 65536 floats (2048.000 MB)
      Column: 8192 floats (32.000 KB)
  Iterations: 100
Average time: 90.35 μs (PASS)
    Accuracy: 0.00 (PASS)

$
```
