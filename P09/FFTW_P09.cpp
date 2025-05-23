//============================================================================
//
// Name        : FFTW_P09.cpp
// Author      : [Your Name/Identifier]
// Version     : [Version Number]
// Copyright   : [Your Copyright Notice]
// Description : Prune noisy tuning fork audio signal using FFTW (CPU) and
//               cuFFT (GPU). Performs Forward FFT -> Filter -> Inverse FFT
//               on both paths. This version times only the FFT transforms
//               and then uses cuBLAS to compute the L2 norm (norm-2) on the
//               GPU final inverse FFT output so that it can be compared with
//               the CPU final values.
//============================================================================

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>          // FFTW library for CPU FFT
#include <cufft.h>          // cuFFT library for GPU FFT
#include <cuda_runtime.h>   // CUDA runtime API
#include <chrono>           // For timing
#include <cublas_v2.h>      // cuBLAS library for norm-2

#include "WavFile.h" // Make sure WavFile.h and WavFile.cpp are present

#define BUFF_SIZE   16384 // Processing buffer size in samples
#define MAX_FREQ    48    // KHz (Used if power analysis is added back)

// Macro for checking CUDA errors
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// --- Helper Function for Filtering (CPU) ---
// This function modifies the complex frequency data in place by zeroing
// out frequency bins around a target frequency.
void filterFrequencyDomain(fftw_complex* data, int N, double sampleRate, double freqToFilter, int /*filterWidth*/) {
    int target_bin = static_cast<int>(freqToFilter * N / sampleRate);
    int mirror_bin = N - target_bin;
    printf("Zeroing bins %d, %d (positive and negative %.2f Hz)\n", target_bin, mirror_bin, freqToFilter);
    if (target_bin >= 0 && target_bin + 1 < N && mirror_bin - 1 >= 0) {
        data[target_bin][0]    = 0.0;
        data[target_bin][1]    = 0.0;
        data[target_bin + 1][0] = 0.0;
        data[target_bin + 1][1] = 0.0;
        data[mirror_bin][0]     = 0.0;
        data[mirror_bin][1]     = 0.0;
        data[mirror_bin - 1][0] = 0.0;
        data[mirror_bin - 1][1] = 0.0;
    }
}

// --- Main Function ---
int main(int argc, char *argv[]) {

    // --- Argument Parsing ---
    const char *wavfile;
    if (argc != 2) {
        fprintf(stderr, "usage: %s <input.wav>\n", argv[0]);
        exit(1);
    } else {
        wavfile = argv[1];
    }

    // --- Output File Naming ---
    char *base_name = strdup(wavfile);
    char *dot = strrchr(base_name, '.');
    if (dot && !strcmp(dot, ".wav")) {
        *dot = '\0';
    }
    char *wavfileout_cpu = (char *)malloc(strlen(base_name) + strlen("_cpu_out.wav") + 1);
    char *wavfileout_gpu = (char *)malloc(strlen(base_name) + strlen("_gpu_out.wav") + 1);
    char *logfile = (char *)malloc(strlen(base_name) + strlen("_out.log") + 1);
    if (!wavfileout_cpu || !wavfileout_gpu || !logfile) {
         fprintf(stderr, "Error allocating memory for filenames.\n");
         free(base_name);
         exit(1);
    }
    sprintf(wavfileout_cpu, "%s_cpu_out.wav", base_name);
    sprintf(wavfileout_gpu, "%s_gpu_out.wav", base_name);
    sprintf(logfile, "%s_out.log", base_name);
    printf("Input WAV file: %s\n", wavfile);
    printf("CPU Output WAV file: %s\n", wavfileout_cpu);
    printf("GPU Output WAV file: %s\n", wavfileout_gpu);
    printf("Log file: %s\n", logfile);
    free(base_name);

    // --- FFTW Setup (Host Memory Allocation) ---
    fftw_complex *h_fft_in, *h_fft_out_cpu, *h_ifft_out_cpu;
    h_fft_in       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    h_fft_out_cpu  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    h_ifft_out_cpu = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    fftw_complex *h_fft_out_gpu_temp;
    h_fft_out_gpu_temp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    if (!h_fft_in || !h_fft_out_cpu || !h_ifft_out_cpu || !h_fft_out_gpu_temp) {
        fprintf(stderr, "Error allocating FFTW host memory.\n");
        fftw_free(h_fft_in); fftw_free(h_fft_out_cpu); fftw_free(h_ifft_out_cpu); fftw_free(h_fft_out_gpu_temp);
        free(wavfileout_cpu); free(wavfileout_gpu); free(logfile);
        exit(1);
    }

    // --- FFTW Plans ---
    fftw_plan plan_forward_cpu = fftw_plan_dft_1d(BUFF_SIZE, h_fft_in, h_fft_out_cpu, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_backward_cpu = fftw_plan_dft_1d(BUFF_SIZE, h_fft_out_cpu, h_ifft_out_cpu, FFTW_BACKWARD, FFTW_ESTIMATE);

    // --- cuFFT Setup (Device Memory Allocation and Plans) ---
    cufftHandle plan_forward_gpu, plan_backward_gpu;
    cufftDoubleComplex *d_fft_data;
    cufftDoubleComplex *h_ifft_out_gpu; // host buffer for final GPU IFFT result
    int nx = BUFF_SIZE;
    int batch = 1;
    cufftType type = CUFFT_Z2Z;
    CUDA_CHECK(cudaMalloc((void**)&d_fft_data, nx * sizeof(cufftDoubleComplex)));
    h_ifft_out_gpu = (cufftDoubleComplex *)calloc(nx, sizeof(cufftDoubleComplex));
    if (!h_ifft_out_gpu) {
        fprintf(stderr, "Error allocating host memory for cuFFT output.\n");
        cudaFree(d_fft_data);
        fftw_free(h_fft_in); fftw_free(h_fft_out_cpu); fftw_free(h_ifft_out_cpu); fftw_free(h_fft_out_gpu_temp);
        fftw_destroy_plan(plan_forward_cpu); fftw_destroy_plan(plan_backward_cpu);
        free(wavfileout_cpu); free(wavfileout_gpu); free(logfile);
        exit(1);
    }
    cufftResult status;
    status = cufftPlan1d(&plan_forward_gpu, nx, type, batch);
    if (status != CUFFT_SUCCESS) {
        printf("error: cufftPlan1d (forward GPU) failed.\n");
        cudaFree(d_fft_data); free(h_ifft_out_gpu);
        fftw_free(h_fft_in); fftw_free(h_fft_out_cpu); fftw_free(h_ifft_out_cpu); fftw_free(h_fft_out_gpu_temp);
        fftw_destroy_plan(plan_forward_cpu); fftw_destroy_plan(plan_backward_cpu);
        free(wavfileout_cpu); free(wavfileout_gpu); free(logfile);
        exit(1);
    }
    status = cufftPlan1d(&plan_backward_gpu, nx, type, batch);
    if (status != CUFFT_SUCCESS) {
        printf("error: cufftPlan1d (backward GPU) failed.\n");
        cufftDestroy(plan_forward_gpu);
        cudaFree(d_fft_data); free(h_ifft_out_gpu);
        fftw_free(h_fft_in); fftw_free(h_fft_out_cpu); fftw_free(h_ifft_out_cpu); fftw_free(h_fft_out_gpu_temp);
        fftw_destroy_plan(plan_forward_cpu); fftw_destroy_plan(plan_backward_cpu);
        free(wavfileout_cpu); free(wavfileout_gpu); free(logfile);
        exit(1);
    }

    // --- Create cuBLAS Handle ---
    cublasHandle_t cublas_handle;
    cublasStatus_t blas_status = cublasCreate(&cublas_handle);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        exit(1);
    }

    // --- Audio File Handling ---
    short sampleBuffer[BUFF_SIZE];       // Buffer to read samples from input WAV
    short outputBufferCpu[BUFF_SIZE];      // Output buffer for CPU path results
    short outputBufferGpu[BUFF_SIZE];      // Output buffer for GPU path results

    WavInFile inFile(wavfile);
    printf("--- Input WAV File Info ---\n");
    printf("SampleRate: %d Hz\n", inFile.getSampleRate());
    printf("BitsPerSample: %d\n", inFile.getNumBits());
    printf("NumChannels: %d\n", inFile.getNumChannels());
    printf("NumSamples: %u\n", inFile.getNumSamples());
    printf("---------------------------\n");

    if (inFile.getNumChannels() != 1) {
        fprintf(stderr, "Error: Input file must be mono.\n");
        cufftDestroy(plan_forward_gpu); cufftDestroy(plan_backward_gpu);
        cudaFree(d_fft_data); free(h_ifft_out_gpu);
        fftw_free(h_fft_in); fftw_free(h_fft_out_cpu); fftw_free(h_ifft_out_cpu); fftw_free(h_fft_out_gpu_temp);
        fftw_destroy_plan(plan_forward_cpu); fftw_destroy_plan(plan_backward_cpu);
        free(wavfileout_cpu); free(wavfileout_gpu); free(logfile);
        exit(1);
    }
    if (inFile.getNumBits() != 16) {
        fprintf(stderr, "Warning: Input file is not 16-bit. Output will be 16-bit.\n");
    }

    // Create output WAV file objects for both CPU and GPU paths
    WavOutFile outFileCpu(wavfileout_cpu, inFile.getSampleRate(), 16, 1);
    WavOutFile outFileGpu(wavfileout_gpu, inFile.getSampleRate(), 16, 1);

    FILE *log_fp;
    if ((log_fp = fopen(logfile, "w")) == NULL) {
        fprintf(stderr, "can't open %s for writing\n", logfile);
        cufftDestroy(plan_forward_gpu); cufftDestroy(plan_backward_gpu);
        cudaFree(d_fft_data); free(h_ifft_out_gpu);
        fftw_free(h_fft_in); fftw_free(h_fft_out_cpu); fftw_free(h_ifft_out_cpu); fftw_free(h_fft_out_gpu_temp);
        fftw_destroy_plan(plan_forward_cpu); fftw_destroy_plan(plan_backward_cpu);
        free(wavfileout_cpu); free(wavfileout_gpu); free(logfile);
        exit(1);
    }

    // --- Timing Variables for FFT Transforms Only ---
    long long total_cpu_fft_duration_us = 0;
    long long total_gpu_fft_duration_us = 0;
    int chunk_count = 0;
    double sampleRate = inFile.getSampleRate();
    double freqToFilter = 10000.0;
    int filterWidth = 2;

    // --- Processing Loop ---
    printf("\nStarting audio processing...\n");
    while (!inFile.eof()) {
        size_t samplesRead = inFile.read(sampleBuffer, BUFF_SIZE);
        if (samplesRead == 0) break;
        chunk_count++;

        // --- Prepare Input Data ---
        for (size_t i = 0; i < BUFF_SIZE; ++i) {
            if (i < samplesRead) {
                h_fft_in[i][0] = (double)sampleBuffer[i];
                h_fft_in[i][1] = 0.0;
            } else {
                h_fft_in[i][0] = 0.0;
                h_fft_in[i][1] = 0.0;
            }
        }

        // === CPU Processing Path (Timing only FFTW transforms) ===
        auto cpu_forward_start = std::chrono::high_resolution_clock::now();
        fftw_execute(plan_forward_cpu);
        auto cpu_forward_end = std::chrono::high_resolution_clock::now();

        // Perform filtering (not timed)
        filterFrequencyDomain(h_fft_out_cpu, BUFF_SIZE, sampleRate, freqToFilter, filterWidth);

        auto cpu_inverse_start = std::chrono::high_resolution_clock::now();
        fftw_execute(plan_backward_cpu);
        auto cpu_inverse_end = std::chrono::high_resolution_clock::now();

        total_cpu_fft_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_forward_end - cpu_forward_start).count();
        total_cpu_fft_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_inverse_end - cpu_inverse_start).count();

        // --- Compute CPU Norm (Normalized by dividing by BUFF_SIZE) ---
        double cpu_norm = 0.0;
        int norm_count = (int)samplesRead; // Only compare for valid samples
        for (int i = 0; i < norm_count; i++) {
            double re = h_ifft_out_cpu[i][0] / BUFF_SIZE;
            double im = h_ifft_out_cpu[i][1] / BUFF_SIZE;
            cpu_norm += re * re + im * im;
        }
        cpu_norm = sqrt(cpu_norm);

        // Prepare CPU Output Buffer (normalize and clamp for output)
        for (size_t i = 0; i < samplesRead; ++i) {
            double real_part = h_ifft_out_cpu[i][0] / BUFF_SIZE;
            if (real_part > 32767.0) real_part = 32767.0;
            else if (real_part < -32768.0) real_part = -32768.0;
            outputBufferCpu[i] = (short)real_part;
        }

        // === GPU Processing Path (Timing only cuFFT transforms) ===
        CUDA_CHECK(cudaMemcpy(d_fft_data, (cufftDoubleComplex*)h_fft_in, nx * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
        auto gpu_forward_start = std::chrono::high_resolution_clock::now();
        status = cufftExecZ2Z(plan_forward_gpu, d_fft_data, d_fft_data, CUFFT_FORWARD);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto gpu_forward_end = std::chrono::high_resolution_clock::now();

        // Copy FFT result to host temporary buffer, filter on CPU, and copy back (not timed)
        CUDA_CHECK(cudaMemcpy((cufftDoubleComplex*)h_fft_out_gpu_temp, d_fft_data, nx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
        filterFrequencyDomain(h_fft_out_gpu_temp, BUFF_SIZE, sampleRate, freqToFilter, filterWidth);
        CUDA_CHECK(cudaMemcpy(d_fft_data, (cufftDoubleComplex*)h_fft_out_gpu_temp, nx * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

        auto gpu_inverse_start = std::chrono::high_resolution_clock::now();
        status = cufftExecZ2Z(plan_backward_gpu, d_fft_data, d_fft_data, CUFFT_INVERSE);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto gpu_inverse_end = std::chrono::high_resolution_clock::now();

        total_gpu_fft_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(gpu_forward_end - gpu_forward_start).count();
        total_gpu_fft_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(gpu_inverse_end - gpu_inverse_start).count();

        // --- Compute GPU Norm using cuBLAS ---
        double gpu_norm = 0.0;
        blas_status = cublasDznrm2(cublas_handle, norm_count, d_fft_data, 1, &gpu_norm);
        if (blas_status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasDznrm2 failed\n");
        }
        // Normalize GPU norm by dividing by BUFF_SIZE (consistent with CPU normalization)
        gpu_norm = gpu_norm / BUFF_SIZE;

        // Print norm comparison for the current chunk
        printf("Chunk %d: CPU norm = %f, GPU norm = %f, Diff = %f\n", chunk_count, cpu_norm, gpu_norm, fabs(cpu_norm - gpu_norm));
        fprintf(log_fp, "Chunk %d: CPU norm = %f, GPU norm = %f, Diff = %f\n", chunk_count, cpu_norm, gpu_norm, fabs(cpu_norm - gpu_norm));

        // Copy final GPU result from Device -> Host (not timed)
        CUDA_CHECK(cudaMemcpy(h_ifft_out_gpu, d_fft_data, nx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

        // Prepare GPU Output Buffer (normalize and clamp for output)
        for (size_t i = 0; i < samplesRead; ++i) {
            double real_part = h_ifft_out_gpu[i].x / nx;
            if (real_part > 32767.0) real_part = 32767.0;
            else if (real_part < -32768.0) real_part = -32768.0;
            outputBufferGpu[i] = (short)real_part;
        }

        // --- Write Outputs ---
        outFileCpu.write(outputBufferCpu, samplesRead);
        outFileGpu.write(outputBufferGpu, samplesRead);
    } // End processing loop

    printf("\nProcessing finished. Processed %d chunks.\n", chunk_count);
    if (chunk_count > 0) {
        double avg_cpu_fft_ms = (double)total_cpu_fft_duration_us / chunk_count / 1000.0;
        double avg_gpu_fft_ms = (double)total_gpu_fft_duration_us / chunk_count / 1000.0;
        printf("\n--- Timing Comparison (Average per %d-sample chunk) ---\n", BUFF_SIZE);
        printf("CPU FFT Transforms (Forward + Inverse): %.4f ms\n", avg_cpu_fft_ms);
        printf("GPU FFT Transforms (Forward + Inverse): %.4f ms\n", avg_gpu_fft_ms);
        printf("--------------------------------------------------------\n");
        fprintf(log_fp, "Processed %d chunks of size %d.\n", chunk_count, BUFF_SIZE);
        fprintf(log_fp, "Filter Target: %.1f Hz, Width: %d bins\n", freqToFilter, filterWidth);
        fprintf(log_fp, "Average CPU FFT time per chunk: %.4f ms\n", avg_cpu_fft_ms);
        fprintf(log_fp, "Average GPU FFT time per chunk: %.4f ms\n", avg_gpu_fft_ms);
        fprintf(log_fp, "CPU output saved to: %s\n", wavfileout_cpu);
        fprintf(log_fp, "GPU output saved to: %s\n", wavfileout_gpu);
    } else {
        printf("\nNo data processed, skipping timing results.\n");
        fprintf(log_fp, "No data processed.\n");
    }

    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    fftw_destroy_plan(plan_forward_cpu);
    fftw_destroy_plan(plan_backward_cpu);
    fftw_free(h_fft_in);
    fftw_free(h_fft_out_cpu);
    fftw_free(h_ifft_out_cpu);
    fftw_free(h_fft_out_gpu_temp);
    cufftDestroy(plan_forward_gpu);
    cufftDestroy(plan_backward_gpu);
    cudaFree(d_fft_data);
    free(h_ifft_out_gpu);
    cublasDestroy(cublas_handle);
    fclose(log_fp);
    free(wavfileout_cpu);
    free(wavfileout_gpu);
    free(logfile);
    printf("Cleanup complete. Exiting.\n");
    return 0;
}
