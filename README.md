# Accelerated Computing

This repository contains my COMPE 596 *Accelerated Computing* programming assignments. Each folder holds the source code (C, C++, CUDA, HIP), a Makefile (or build script), and any supporting bash scripts to compile and run the assignment.

## Assignments

- **P01** – **OpenMP Matrix Multiplication**  
  - C program that multiplies two \(N\times N\) matrices in parallel using OpenMP  
  - Includes a Makefile for building and a bash script to run benchmarks over 1–128 threads and print timing results  

- **P02** – **Parallel Doubly-Linked List Insertion**  
  - C implementation of sorted-list insertion with hand-over-hand locking in OpenMP  
  - Benchmarks insertion time vs. thread count and list size  

- **P03** – **OpenMP Simpson’s-Rule Integration**  
  - C program to approximate `$ \displaystyle \int_{0}^{\pi/2} \frac{\arccos(\cos x)}{1 + 2\cos x} \, dx $`
     using Simpson’s rule in parallel  
  - Build and run scripts measure runtime and error for different thread/partition counts  

- **P04** – **Jacobi 2D Solver (CPU & CUDA)**  
  - Hybrid C/CUDA code implementing the Jacobi iterative method on a 2D grid  
  - Compares CPU serial, CPU parallel (OpenMP), GPU non-SIMD, and GPU SIMD kernels  
  - Makefile and bash scripts to build and run each variant  

- **P05** – **CUDA Reduction for Array Summation**  
  - CUDA kernel for parallel reduction to sum large arrays  
  - Serial C version included for baseline comparison  
  - Build scripts automate array-size sweeps and print speedup metrics  

- **P06** – **Naïve vs Tiled GPU Matrix Multiplication**  
  - Two CUDA kernels: naïve and tiled (16×16 shared-memory) for multiplying \(M\times1024\) by \(1024\times M\)  
  - Makefile builds both versions and a runner script measures relative performance  

- **P07** – **Sobel 5×5 Convolution in CUDA**  
  - CUDA kernel applying a 5×5 horizontal Sobel filter to an image buffer  
  - Includes a minimal C driver and build scripts to compile and execute on test data  

- **P08** – **cuSOLVER LU Factorization & Solve**  
  - C++ program using cuSOLVER to factor and solve Hilbert systems of size \(2^1\)–\(2^{10}\)  
  - Scripts to build, run, perturb right-hand sides, and print solver timings  

- **P09** – **GPU-Accelerated Audio Filtering (cuFFT vs FFTW)**  
  - C++/CUDA code that reads a WAV file, zeroes out a 10 kHz tone in the frequency domain via FFT  
  - Comparison between FFTW (CPU) and cuFFT (GPU) implementations  
  - Makefile and run script to build and filter test audio  

- **P11** – **Dense vs Sparse GEMM with ROCm**  
  - HIP program timing dense GEMM (rocBLAS) vs. sparse GEMM (rocSPARSE) on large matrices  
  - Build scripts automate runs at various sparsity levels  

---

**Note:** P10 (OpenCL) was omitted. Each assignment folder contains all source files, build instructions, and scripts needed to reproduce the results. Feel free to clone and explore!  

