#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>

#define TILE_WIDTH 16

// CUDA kernel for 2D convolution over RGB image
__global__
void convolution(unsigned int *in, int *mask, int *out,
                 int channels, int width, int height)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < width && Row < height) {
        for (int c = 0; c < channels; ++c) {
            int pixVal = 0;
            int N_start_col = Col - 2;  // 5x5 kernel: 5 / 2 = 2
            int N_start_row = Row - 2;

            for (int j = 0; j < 5; ++j) {
                for (int k = 0; k < 5; ++k) {
                    int curRow = N_start_row + j;
                    int curCol = N_start_col + k;

                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int imageIdx = (curRow * width + curCol) * channels + c;
                        int maskIdx = j * 5 + k;
                        pixVal += in[imageIdx] * mask[maskIdx];
                    }
                }
            }

            int outIdx = (Row * width + Col) * channels + c;
            out[outIdx] = pixVal;
        }
    }
}

int main(int argc, char *argv[])
{
    unsigned int *hostInputImage;
    int *hostOutputImage;
    unsigned int inputLength = 589824; // 384 * 512 * 3 = 589824

    printf("%% Importing 3-channel image data and creating memory on host\n");

    hostInputImage = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
    hostOutputImage = (int *)malloc(inputLength * sizeof(int));

    FILE *f;
    unsigned int pixelValue, i = 0;
    f = fopen("peppers.dat", "r");
    if (!f) {
        printf("Error opening input file.\n");
        return 1;
    }
    while (!feof(f) && i < inputLength) {
        fscanf(f, "%d", &pixelValue);
        hostInputImage[i++] = pixelValue;
    }
    fclose(f);

    int maskRows = 5;
    int maskColumns = 5;
    int imageChannels = 3;
    int imageWidth = 512;
    int imageHeight = 384;

    // Sobel 5x5 horizontal convolution kernel for edge detection
    int hostMask[5][5] = {
        {2,  2,  4,  2,  2},
        {1,  1,  2,  1,  1},
        {0,  0,  0,  0,  0},
        {-1, -1, -2, -1, -1},
        {-2, -2, -4, -2, -2}
    };

    unsigned int *deviceInputImage;
    int *deviceOutputImage;
    int *deviceMask;

    assert(maskRows == 5);
    assert(maskColumns == 5);

    cudaMalloc((void **)&deviceInputImage,
               imageWidth * imageHeight * imageChannels * sizeof(unsigned int));

    cudaMalloc((void **)&deviceOutputImage,
               imageWidth * imageHeight * imageChannels * sizeof(int));

    cudaMalloc((void **)&deviceMask, maskRows * maskColumns * sizeof(int));

    cudaMemcpy(deviceInputImage,
               hostInputImage,
               imageWidth * imageHeight * imageChannels * sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(deviceMask,
               hostMask,
               maskRows * maskColumns * sizeof(int),
               cudaMemcpyHostToDevice);

    dim3 dimGrid((int)ceil((float)imageWidth / TILE_WIDTH),
                 (int)ceil((float)imageHeight / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    convolution<<<dimGrid, dimBlock>>>(deviceInputImage,
                                       deviceMask,
                                       deviceOutputImage,
                                       imageChannels, imageWidth, imageHeight);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error during device sync: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hostOutputImage,
               deviceOutputImage,
               imageWidth * imageHeight * imageChannels * sizeof(int),
               cudaMemcpyDeviceToHost);

    f = fopen("peppers.out", "w");
    if (!f) {
        printf("Error opening output file.\n");
        return 1;
    }
    for (int i = 0; i < inputLength; ++i)
        fprintf(f, "%d\n", hostOutputImage[i]);
    fclose(f);

    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
    cudaFree(deviceMask);
    free(hostInputImage);
    free(hostOutputImage);

    printf("Convolution completed. Output written to peppers.out\n");
    return 0;
}
