#!/bin/bash

# Change to the directory
cd /mnt/beegfs/dgx/jrathi/COMPE596/P04 || { echo "Directory not found!"; exit 1; }

# Load necessary modules (if required for compilation)
# module load gcc   # Uncomment if your system requires it

# Compile the C program
gcc -o serial serial.c -lm

# Run the program and log output
nohup ./serial > jacobi_output.log 2>&1 &

echo "Jacobi iteration started in the background. Check jacobi_output.log for progress."
