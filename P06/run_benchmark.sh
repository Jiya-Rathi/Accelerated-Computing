#!/bin/bash

nvcc -o matrix_mul matrix_mul.cu

for ((exp=5; exp<=15; exp++)); do
    size=$((2**exp))
    echo "Running matrix size: A=${size}x1024, B=1024x${size}, C=${size}x${size}"
    ./matrix_mul $size
done
