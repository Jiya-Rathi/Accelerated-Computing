// dense_gemm_8192_timed.cpp
// HIP/C++ + rocBLAS example for 2^13 x 2^13 dense GEMM with timing

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Error-checking macros
#define CHECK_HIP(expr)                                          \
    do {                                                         \
        hipError_t _err = (expr);                               \
        if (_err != hipSuccess) {                               \
            fprintf(stderr,                                     \
                    "HIP error %s:%d: '%s'\n",                  \
                    __FILE__, __LINE__, hipGetErrorString(_err)); \
            std::exit(EXIT_FAILURE);                            \
        }                                                        \
    } while(0)

#define CHECK_ROCBLAS(expr)                                      \
    do {                                                         \
        rocblas_status _st = (expr);                             \
        if (_st != rocblas_status_success) {                     \
            fprintf(stderr,                                     \
                    "rocBLAS error %s:%d: %d\n",                \
                    __FILE__, __LINE__, (int)_st);               \
            std::exit(EXIT_FAILURE);                            \
        }                                                        \
    } while(0)

int main()
{
    // Matrix size: 2^13 x 2^13
    const int N = 1 << 13;   // 8192
    const int M = N;
    const int K = N;

    // Leading dimensions (column-major)
    const int lda = M;
    const int ldb = K;
    const int ldc = M;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Host allocations (~800 MB each)
    std::vector<float> hA(size_t(lda) * K, 1.0f);
    std::vector<float> hB(size_t(ldb) * N, 1.0f);
    std::vector<float> hC(size_t(ldc) * N);

    // Device pointers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // Allocate GPU memory
    CHECK_HIP( hipMalloc(&dA, hA.size() * sizeof(float)) );
    CHECK_HIP( hipMalloc(&dB, hB.size() * sizeof(float)) );
    CHECK_HIP( hipMalloc(&dC, hC.size() * sizeof(float)) );

    // Copy A and B to device
    CHECK_HIP( hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice) );
    CHECK_HIP( hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice) );

    // Create rocBLAS handle
    rocblas_handle handle;
    CHECK_ROCBLAS( rocblas_create_handle(&handle) );

    // Create HIP events for timing
    hipEvent_t start, stop;
    CHECK_HIP( hipEventCreate(&start) );
    CHECK_HIP( hipEventCreate(&stop) );

    // Record start event
    CHECK_HIP( hipEventRecord(start, nullptr) );

    // Perform C = alpha * A * B + beta * C on GPU
    CHECK_ROCBLAS(
        rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            M, N, K,
            &alpha,
            dA, lda,
            dB, ldb,
            &beta,
            dC, ldc
        )
    );

    // Record stop event and synchronize
    CHECK_HIP( hipEventRecord(stop, nullptr) );
    CHECK_HIP( hipEventSynchronize(stop) );

    // Compute elapsed time (ms)
    float elapsed_ms = 0.0f;
    CHECK_HIP( hipEventElapsedTime(&elapsed_ms, start, stop) );
    printf("rocBLAS sgemm time: %.3f ms\n", elapsed_ms);

    // Copy result back to host
    CHECK_HIP( hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost) );

    // Quick correctness check (first 10 entries should be K)
    bool ok = true;
    for(int i = 0; i < 10; ++i) {
        if (std::fabs(hC[i] - float(K)) > 1e-3f) {
            fprintf(stderr, "Mismatch at idx %d: got %.3f, expected %.3f\n", i, hC[i], float(K));
            ok = false;
            break;
        }
    }
    printf("Result verification: %s\n", ok ? "PASS" : "FAIL");

    // Clean up
    hipEventDestroy(start);
    hipEventDestroy(stop);
    rocblas_destroy_handle(handle);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
