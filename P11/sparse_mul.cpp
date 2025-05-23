#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>

// Error-checking macros
#define CHECK_HIP(expr)                                          \
    do {                                                         \
        hipError_t _err = (expr);                               \
        if (_err != hipSuccess) {                               \
            fprintf(stderr, "HIP error %s:%d: '%s'\n",       \
                    __FILE__, __LINE__, hipGetErrorString(_err));\
            std::exit(EXIT_FAILURE);                            \
        }                                                        \
    } while(0)

#define CHECK_ROCSPARSE(expr)                                    \
    do {                                                         \
        rocsparse_status _st = (expr);                           \
        if (_st != rocsparse_status_success) {                   \
            fprintf(stderr, "rocSPARSE error %s:%d: %d\n",    \
                    __FILE__, __LINE__, (int)_st);               \
            std::exit(EXIT_FAILURE);                            \
        }                                                        \
    } while(0)

float rand01() { return static_cast<float>(rand()) / RAND_MAX; }

int main()
{
    const int N = 1 << 13;  // 8192
    const int M = N;
    const int K = N;
    const float density = 0.01f; // adjust as needed
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Host dense A and B
    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);
    // initialize random sparse A and dense B
    for(int i = 0; i < M*K; ++i)
        hA[i] = (rand01() < density ? rand01() : 0.0f);
    for(int i = 0; i < K*N; ++i)
        hB[i] = rand01();

    // Convert A to CSR
    std::vector<int>   hRowPtr; hRowPtr.reserve(M+1);
    std::vector<int>   hColInd; hColInd.reserve(M*K*density);
    std::vector<float> hVal;    hVal.reserve(M*K*density);
    int ptr = 0;
    for(int i = 0; i < M; ++i) {
        hRowPtr.push_back(ptr);
        for(int j = 0; j < K; ++j) {
            float v = hA[i*K + j];
            if(v != 0.0f) {
                hVal.push_back(v);
                hColInd.push_back(j);
                ++ptr;
            }
        }
    }
    hRowPtr.push_back(ptr);
    int nnz = ptr;

    // Device buffers
    float *dVal = nullptr, *dB = nullptr, *dC = nullptr;
    int   *dRowPtr = nullptr, *dColInd = nullptr;

    CHECK_HIP( hipMalloc(&dVal,    sizeof(float) * nnz) );
    CHECK_HIP( hipMalloc(&dColInd, sizeof(int)   * nnz) );
    CHECK_HIP( hipMalloc(&dRowPtr, sizeof(int)   * (M+1)) );
    CHECK_HIP( hipMalloc(&dB,      sizeof(float) * K * N) );
    CHECK_HIP( hipMalloc(&dC,      sizeof(float) * M * N) );

    CHECK_HIP( hipMemcpy(dVal,    hVal.data(),    nnz * sizeof(float), hipMemcpyHostToDevice) );
    CHECK_HIP( hipMemcpy(dColInd, hColInd.data(), nnz * sizeof(int),   hipMemcpyHostToDevice) );
    CHECK_HIP( hipMemcpy(dRowPtr, hRowPtr.data(), (M+1) * sizeof(int), hipMemcpyHostToDevice) );
    CHECK_HIP( hipMemcpy(dB,      hB.data(),      K*N * sizeof(float), hipMemcpyHostToDevice) );

    // Create rocSPARSE handle & descriptor
    rocsparse_handle     handle;
    rocsparse_mat_descr  descr;
    CHECK_ROCSPARSE( rocsparse_create_handle(&handle) );
    CHECK_ROCSPARSE( rocsparse_create_mat_descr(&descr) );

    // Timing
    hipEvent_t start, stop;
    CHECK_HIP( hipEventCreate(&start) );
    CHECK_HIP( hipEventCreate(&stop) );

    // Launch
    CHECK_HIP( hipEventRecord(start, nullptr) );
    CHECK_ROCSPARSE(
        rocsparse_scsrmm(
            handle,
            rocsparse_operation_none,
            rocsparse_operation_none,
            M, N, K,
            nnz,
            &alpha,
            descr,
            dVal,
            dRowPtr,
            dColInd,
            dB, K,
            &beta,
            dC, M
        )
    );
    CHECK_HIP( hipEventRecord(stop, nullptr) );
    CHECK_HIP( hipEventSynchronize(stop) );

    float elapsed_ms = 0.0f;
    CHECK_HIP( hipEventElapsedTime(&elapsed_ms, start, stop) );
    printf("rocSPARSE scsrmm time: %.3f ms\n", elapsed_ms);

    // Cleanup
    rocsparse_destroy_mat_descr(descr);
    rocsparse_destroy_handle(handle);
    hipFree(dVal);
    hipFree(dColInd);
    hipFree(dRowPtr);
    hipFree(dB);
    hipFree(dC);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
