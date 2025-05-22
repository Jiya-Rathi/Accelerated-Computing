#!/bin/bash

# Change to the directory
cd /mnt/beegfs/dgx/jrathi/COMPE596/P04/gpu_off/parallel || { echo "Directory not found!"; exit 1; }

# Load necessary modules (if required for compilation)
# module load gcc   # Uncomment if your system requires it

# Compile the C program
nvc -mp=gpu -gpu=cc80 -o jiya_parallel_gpu jiya_parallel_gpu.c -lm

# Run the program and log output
nohup ./jiya_parallel_gpu > jacobi_output.log 2>&1 &

echo "Jacobi iteration started in the background. Check jacobi_output.log for progress."
