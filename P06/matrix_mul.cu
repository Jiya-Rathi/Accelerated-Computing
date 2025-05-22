#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_WIDTH 32
#define FIXED_P 1024  // shared dimension

__global__ void matrixMultiply(const float* A, const float* B, float* C,
                               int m, int p, int pB, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < p; k++) {
            sum += A[row * p + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrixTiledMultiply(const float* A, const float* B, float* C,
                                    int m, int p, int pB, int n)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    int numTiles = (p + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int tiledACol = t * TILE_WIDTH + threadIdx.x;
        int tiledBRow = t * TILE_WIDTH + threadIdx.y;

        if (row < m && tiledACol < p)
            tileA[threadIdx.y][threadIdx.x] = A[row * p + tiledACol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && tiledBRow < p)
            tileB[threadIdx.y][threadIdx.x] = B[tiledBRow * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            val += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = val;
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s <m_size>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int p = FIXED_P;
    int n = m;

    printf("Matrix A: %dx%d, B: %dx%d, C: %dx%d\n", m, p, p, n, m, n);

    size_t sizeA = m * p * sizeof(float);
    size_t sizeB = p * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    float *hA = (float*)malloc(sizeA);
    float *hB = (float*)malloc(sizeB);
    float *hC = (float*)malloc(sizeC);
    float *hC_tiled = (float*)malloc(sizeC);

    if (!hA || !hB || !hC || !hC_tiled) {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    srand(0);
    for (int i = 0; i < m * p; i++) hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < p * n; i++) hB[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    if (cudaMalloc(&dA, sizeA) != cudaSuccess ||
        cudaMalloc(&dB, sizeB) != cudaSuccess ||
        cudaMalloc(&dC, sizeC) != cudaSuccess) {
        printf("Device memory allocation failed — skipping size m=%d\n", m);
        return 1;
    }

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
              (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Naive
    cudaMemset(dC, 0, sizeC);
    cudaEventRecord(start);
    matrixMultiply<<<grid, block>>>(dA, dB, dC, m, p, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naiveMs = 0.0f;
    cudaEventElapsedTime(&naiveMs, start, stop);
    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    // Tiled
    cudaMemset(dC, 0, sizeC);
    cudaEventRecord(start);
    matrixTiledMultiply<<<grid, block>>>(dA, dB, dC, m, p, p, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiledMs = 0.0f;
    cudaEventElapsedTime(&tiledMs, start, stop);
    cudaMemcpy(hC_tiled, dC, sizeC, cudaMemcpyDeviceToHost);

    printf("  Naive kernel time: %.3f ms\n", naiveMs);
    printf("  Tiled kernel time: %.3f ms\n\n", tiledMs);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(hA); free(hB); free(hC); free(hC_tiled);

    return 0;
}
