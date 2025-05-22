#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <assert.h>
#include <chrono>
#include <time.h>

#define PERTURB_B_VECTOR 1
#define PRINT_SOLUTION_SAMPLE 1

inline void checkCuda(cudaError_t result, char const *const func, const int line)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                __FILE__, line, static_cast<unsigned int>(result),
                cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(val) checkCuda((val), #val, __LINE__)

inline void checkCusolver(cusolverStatus_t status, char const *const func, const int line)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSolver error at %s:%d code=%d \n",
                __FILE__, line, static_cast<int>(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUSOLVER(val) checkCusolver((val), #val, __LINE__)

void buildHilbertMatrix(double* A, int N)
{
    for(int col = 0; col < N; col++) {
        for(int row = 0; row < N; row++) {
            A[row + col*N] = 1.0 / (double)(row + col + 1);
        }
    }
}

void buildBVector(double* B, int N)
{
    for(int i = 0; i < N; i++) {
        B[i] = 1.0;
    }
}

void perturbBVector(double* B, int N)
{
    for(int i = 0; i < N; i++) {
        double eps = (double)rand() / (double)RAND_MAX;
        B[i] += eps;
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));

    cusolverDnHandle_t cusolverH = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    const int pivot_on = 0;

    int maxPower = 10;
    if (argc > 1) {
        maxPower = atoi(argv[1]);
    }

    printf("%-6s | %-25s | %-20s\n", "N", "Factor + Solve Time (ms)", "Solve Only Time (ms)");
    printf("--------------------------------------------------------------\n");

    for(int power = 1; power <= maxPower; power++) {
        int N = 1 << power;

        size_t sizeA = N * N * sizeof(double);
        size_t sizeB = N * sizeof(double);

        double* h_A = (double*) malloc(sizeA);
        double* h_B = (double*) malloc(sizeB);
        double* h_X = (double*) malloc(sizeB);

        buildHilbertMatrix(h_A, N);
        buildBVector(h_B, N);

        double *d_A = nullptr, *d_B = nullptr;
        int *d_info = nullptr;
        CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
        CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
        CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

        int lwork = 0;
        CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cusolverH, N, N, d_A, N, &lwork));

        double* d_work = nullptr;
        CHECK_CUDA(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));

        CHECK_CUSOLVER(cusolverDnDgetrf(cusolverH, N, N, d_A, N, d_work, nullptr, d_info));
        CHECK_CUSOLVER(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_A, N, nullptr, d_B, N, d_info));

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_factor_solve = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_factor_solve, start, stop));
        CHECK_CUDA(cudaMemcpy(h_X, d_B, sizeB, cudaMemcpyDeviceToHost));

#if PRINT_SOLUTION_SAMPLE
        printf("N = %4d | x[0] = %.6f, x[N/2] = %.6f, x[N-1] = %.6f\n", N, h_X[0], h_X[N/2], h_X[N-1]);
#endif

#if PERTURB_B_VECTOR
        buildBVector(h_B, N);
        perturbBVector(h_B, N);
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
#endif

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUSOLVER(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_A, N, nullptr, d_B, N, d_info));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_solve_only = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_solve_only, start, stop));

        printf("%-6d | %-25f | %-20f\n", N, ms_factor_solve, ms_solve_only);

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_info));
        CHECK_CUDA(cudaFree(d_work));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        free(h_A);
        free(h_B);
        free(h_X);
    }

    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    cudaDeviceReset();
    return 0;
}
