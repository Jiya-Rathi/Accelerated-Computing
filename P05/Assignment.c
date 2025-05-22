#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

__global__ void reduction(float *input, float *output, int len) {
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
    
    if (start + t < len)
        partialSum[t] = input[start + t];
    else
        partialSum[t] = 0;
    
    if (start + BLOCK_SIZE + t < len)
        partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
        partialSum[BLOCK_SIZE + t] = 0;
    
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    
    if (t == 0)
        output[blockIdx.x] = partialSum[0];
}

int main(int argc, char *argv[]) {
    int lower_power = 12, upper_power = 24;
    if (argc >= 3) {
        lower_power = atoi(argv[1]);
        upper_power = atoi(argv[2]);
    }
    
    for (int power = lower_power; power <= upper_power; power++) {
        int N = 1 << power;
        size_t bytes = N * sizeof(float);
        
        // CPU Summation
        int *array = (int *)calloc(N, sizeof(int));
        for (int i = 0; i < N; i++) array[i] = 1;
        clock_t start = clock();
        int sum = 0;
        for (int i = 0; i < N; i++) sum += array[i];
        clock_t end = clock();
        double cpuTime = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        free(array);
        
        // GPU Summation
        float *h_in = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;
        float *d_in, *d_out;
        cudaMalloc((void**)&d_in, bytes);
        cudaMalloc((void**)&d_out, bytes);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        
        cudaEvent_t gpuStart, gpuStop;
        cudaEventCreate(&gpuStart);
        cudaEventCreate(&gpuStop);
        cudaEventRecord(gpuStart);
        
        int numElements = N, gridSize;
        while (numElements > 1) {
            gridSize = (numElements + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
            reduction<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, numElements);
            cudaMemcpy(d_in, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToDevice);
            numElements = gridSize;
        }
        
        cudaEventRecord(gpuStop);
        cudaEventSynchronize(gpuStop);
        float gpuTime;
        cudaEventElapsedTime(&gpuTime, gpuStart, gpuStop);
        
        float gpuSum;
        cudaMemcpy(&gpuSum, d_in, sizeof(float), cudaMemcpyDeviceToHost);
        
        double speedup = cpuTime / gpuTime;
        printf("Array Size: 2^%d = %d, CPU Time: %.4f ms, GPU Time: %.4f ms, Speedup: %.2f\n",
               power, N, cpuTime, gpuTime, speedup);
        
        free(h_in);
        cudaFree(d_in);
        cudaFree(d_out);
        cudaEventDestroy(gpuStart);
        cudaEventDestroy(gpuStop);
    }
    return 0;
}

	

