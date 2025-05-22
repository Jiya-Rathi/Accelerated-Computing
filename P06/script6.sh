#!/bin/bash

# Compile the CUDA code
nvcc -o assig6 assig6.cu

# Loop through powers of 2 from 2^10 to 2^20
for ((exp=5; exp<=15; exp++)); do
    size=$((2**exp))
    echo "Running matrix size: $size x $size"
    ./assig6 $size
done
