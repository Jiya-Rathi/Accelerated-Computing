#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

// Error-checking macros
#define CHECK_HIP(expr) do { hipError_t err_ = (expr); if(err_ != hipSuccess) { std::cerr << "HIP error " << hipGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE);} } while(0)
#define CHECK_ROCBLAS(expr) do { rocblas_status st_ = (expr); if(st_ != rocblas_status_success) { std::cerr << "rocBLAS error " << st_ << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE);} } while(0)
#define CHECK_ROCSPARSE(expr) do { rocsparse_status st_ = (expr); if(st_ != rocsparse_status_success) { std::cerr << "rocSPARSE error " << st_ << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE);} } while(0)

static float rand01() { return static_cast<float>(rand()) / RAND_MAX; }

int main()
{
    const int M = 1 << 13;
    const int N = 1 << 13;
    const int K = 1 << 13;
    const std::vector<float> densities = {1e-5f,1e-4f,1e-3f,1e-2f,1e-1f,1e0f};

    // rocBLAS & rocSPARSE setup
    rocblas_handle blas_handle;
    CHECK_ROCBLAS( rocblas_create_handle(&blas_handle) );
    rocsparse_handle sparse_handle;
    CHECK_ROCSPARSE( rocsparse_create_handle(&sparse_handle) );
    rocsparse_mat_descr descrA;
    CHECK_ROCSPARSE( rocsparse_create_mat_descr(&descrA) );

    // HIP events
    hipEvent_t start, stop;
    CHECK_HIP( hipEventCreate(&start) );
    CHECK_HIP( hipEventCreate(&stop) );

    // Pre-allocate device buffers
    float *dAd = nullptr, *dBd = nullptr, *dCd = nullptr;
    CHECK_HIP( hipMalloc(&dAd, M*K*sizeof(float)) );
    CHECK_HIP( hipMalloc(&dBd, K*N*sizeof(float)) );
    CHECK_HIP( hipMalloc(&dCd, M*N*sizeof(float)) );

    int *dRowA = nullptr, *dColA = nullptr;
    float *dValA = nullptr, *dB2 = nullptr, *dC2 = nullptr;
    // worst-case nnz = M*K at density=1.0
    size_t max_nnz = size_t(M)*K;
    CHECK_HIP( hipMalloc(&dRowA, (M+1)*sizeof(int)) );
    CHECK_HIP( hipMalloc(&dColA, max_nnz*sizeof(int)) );
    CHECK_HIP( hipMalloc(&dValA, max_nnz*sizeof(float)) );
    CHECK_HIP( hipMalloc(&dB2, K*N*sizeof(float)) );
    CHECK_HIP( hipMalloc(&dC2, M*N*sizeof(float)) );

    std::cout<<"density,ms_dense,ms_sparse"<<std::endl;
    srand(123);

    for(float density: densities)
    {
        // generate host matrices
        std::vector<float> hA(M*K), hB(K*N);
        for(size_t i=0;i<hA.size();++i) hA[i] = (rand01()<density?rand01():0.0f);
        for(size_t i=0;i<hB.size();++i) hB[i] = rand01();

        // copy dense A,B
        CHECK_HIP( hipMemcpy(dAd, hA.data(), hA.size()*sizeof(float), hipMemcpyHostToDevice) );
        CHECK_HIP( hipMemcpy(dBd, hB.data(), hB.size()*sizeof(float), hipMemcpyHostToDevice) );

        // time dense GEMM only
        const float alpha=1.0f, beta=0.0f;
        CHECK_HIP( hipEventRecord(start,nullptr) );
        CHECK_ROCBLAS( rocblas_sgemm(blas_handle, rocblas_operation_none, rocblas_operation_none,
                                     M, N, K, &alpha, dAd, M, dBd, K, &beta, dCd, M) );
        CHECK_HIP( hipEventRecord(stop,nullptr) );
        CHECK_HIP( hipEventSynchronize(stop) );
        float ms_dense=0;
        CHECK_HIP( hipEventElapsedTime(&ms_dense,start,stop) );

        // build CSR for A
        std::vector<int> hRowPtrA(M+1); hRowPtrA[0]=0;
        std::vector<int> hColA; hColA.reserve(size_t(density*M*K));
        std::vector<float> hValA; hValA.reserve(size_t(density*M*K));
        for(int i=0;i<M;++i){
            int row_nnz=0;
            for(int j=0;j<K;++j){ float v=hA[i*K+j]; if(v!=0.0f){hValA.push_back(v);hColA.push_back(j);row_nnz++;}}
            hRowPtrA[i+1]=hRowPtrA[i]+row_nnz;
        }
        int nnzA = hValA.size();

        // copy sparse A and dense B
        CHECK_HIP( hipMemcpy(dRowA,hRowPtrA.data(),(M+1)*sizeof(int),hipMemcpyHostToDevice) );
        CHECK_HIP( hipMemcpy(dColA,hColA.data(),nnzA*sizeof(int),hipMemcpyHostToDevice) );
        CHECK_HIP( hipMemcpy(dValA,hValA.data(),nnzA*sizeof(float),hipMemcpyHostToDevice) );
        CHECK_HIP( hipMemcpy(dB2,hB.data(),hB.size()*sizeof(float),hipMemcpyHostToDevice) );

        // time sparse GEMM only
        CHECK_HIP( hipEventRecord(start,nullptr) );
        CHECK_ROCSPARSE( rocsparse_scsrmm(sparse_handle, rocsparse_operation_none, rocsparse_operation_none,
                                         M, N, K, nnzA, &alpha, descrA, dValA, dRowA, dColA, dB2, K, &beta, dC2, M) );
        CHECK_HIP( hipEventRecord(stop,nullptr) );
        CHECK_HIP( hipEventSynchronize(stop) );
        float ms_sparse=0;
        CHECK_HIP( hipEventElapsedTime(&ms_sparse,start,stop) );

        std::cout<<density<<","<<ms_dense<<","<<ms_sparse<<std::endl;
    }

    // cleanup
    hipFree(dAd); hipFree(dBd); hipFree(dCd);
    hipFree(dRowA); hipFree(dColA); hipFree(dValA); hipFree(dB2); hipFree(dC2);
    hipEventDestroy(start); hipEventDestroy(stop);
    rocblas_destroy_handle(blas_handle);
    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(sparse_handle);
    return 0;
}
