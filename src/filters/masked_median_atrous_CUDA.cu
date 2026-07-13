#include "filters.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

struct PixelLuminance {
    float lum;
    float r, g, b;
};

// Masked median filter kernel: detects salt and pepper noise (impulsive outliers)
// and applies median filtering only at those locations, leaving other pixels untouched.
__global__ void maskedMedianFilterKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width, const int height,
    const int kernelSize, const float threshold)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int plane = width * height;
        const int centerIdx = y * width + x;

        // Support kernel size up to 5x5 (25 elements)
        PixelLuminance neighbors[25];
        int count = 0;
        const int radius = kernelSize / 2;

        for (int dy = -radius; dy <= radius; ++dy) {
            int qy = y + dy;
            qy = (qy < 0) ? 0 : ((qy >= height) ? height - 1 : qy);
            for (int dx = -radius; dx <= radius; ++dx) {
                int qx = x + dx;
                qx = (qx < 0) ? 0 : ((qx >= width) ? width - 1 : qx);

                const int qIdx = qy * width + qx;
                float r = input[qIdx];
                float g = input[plane + qIdx];
                float b = input[2 * plane + qIdx];
                float lum = 0.299f * r + 0.587f * g + 0.114f * b;

                neighbors[count++] = {lum, r, g, b};
            }
        }

        // Center pixel index in the loaded neighbors array
        const int centerNeighborIdx = radius * kernelSize + radius;
        const float center_r = neighbors[centerNeighborIdx].r;
        const float center_g = neighbors[centerNeighborIdx].g;
        const float center_b = neighbors[centerNeighborIdx].b;
        const float center_lum = neighbors[centerNeighborIdx].lum;

        // Find min and max of neighbors (excluding the center pixel itself)
        float min_lum = 1e10f;
        float max_lum = -1e10f;
        for (int i = 0; i < count; ++i) {
            if (i == centerNeighborIdx) continue;
            min_lum = fminf(min_lum, neighbors[i].lum);
            max_lum = fmaxf(max_lum, neighbors[i].lum);
        }

        // Sort neighbors using simple insertion sort to find the median
        for (int i = 1; i < count; ++i) {
            PixelLuminance key = neighbors[i];
            int j = i - 1;
            while (j >= 0 && neighbors[j].lum > key.lum) {
                neighbors[j + 1] = neighbors[j];
                j = j - 1;
            }
            neighbors[j + 1] = key;
        }

        const int medianIdx = count / 2;
        const float median_lum = neighbors[medianIdx].lum;
        const float median_r = neighbors[medianIdx].r;
        const float median_g = neighbors[medianIdx].g;
        const float median_b = neighbors[medianIdx].b;

        // An outlier is defined as being a local extremum (outside neighbor range)
        // and deviating from the median by more than the threshold.
        bool is_outlier = (center_lum <= min_lum || center_lum >= max_lum) && 
                          (fabsf(center_lum - median_lum) > threshold);

        if (is_outlier) {
            output[centerIdx] = median_r;
            output[plane + centerIdx] = median_g;
            output[2 * plane + centerIdx] = median_b;
        } else {
            output[centerIdx] = center_r;
            output[plane + centerIdx] = center_g;
            output[2 * plane + centerIdx] = center_b;
        }
    }
}

namespace Filters {

Image maskedMedianAtrousCUDA(const Image& input, const Image& normal, const Image& albedo,
                             int passes, float sigmaColor, float sigmaNormal, float sigmaAlbedo,
                             int medianKernelSize, float medianThreshold)
{
    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    const int totalPixels = width * height * channels;
    size_t imgSize = totalPixels * sizeof(float);

    Image output(width, height, channels);

    float* d_input = nullptr;
    float* d_normal = nullptr;
    float* d_albedo = nullptr;
    float* d_output = nullptr;
    float* d_temp_median = nullptr;
    float* d_temp1 = nullptr;
    float* d_temp2 = nullptr;

    // Allocate GPU memory
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_normal, imgSize);
    cudaMalloc(&d_albedo, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_temp_median, imgSize);
    cudaMalloc(&d_temp1, imgSize);
    cudaMalloc(&d_temp2, imgSize);

    // Copy input data from host to device
    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, normal.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo, albedo.getData(), imgSize, cudaMemcpyHostToDevice);

    // Run masked median pre-filtering pass
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Clamp median kernel size to valid range [3, 5]
    int mK = medianKernelSize;
    if (mK != 3 && mK != 5) {
        mK = (mK < 4) ? 3 : 5;
    }

    maskedMedianFilterKernel<<<grid, threads>>>(d_input, d_temp_median, width, height, mK, medianThreshold);

    // Run À-Trous Wavelet filter on the outlier-corrected image
    aTrousWaveletCUDA_NoAlloc(d_temp_median, d_normal, d_albedo, d_output, d_temp1, d_temp2,
                              width, height, channels, passes, sigmaColor, sigmaNormal, sigmaAlbedo);

    // Copy result back to host
    cudaMemcpy(output.getData(), d_output, imgSize, cudaMemcpyDeviceToHost);

    // Cleanup GPU resources
    cudaFree(d_input);
    cudaFree(d_normal);
    cudaFree(d_albedo);
    cudaFree(d_output);
    cudaFree(d_temp_median);
    cudaFree(d_temp1);
    cudaFree(d_temp2);

    return output;
}

} // namespace Filters
