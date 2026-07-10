#include "filters.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>

#define MAX_KERNEL_SIZE 129
__constant__ float c_kernel[MAX_KERNEL_SIZE];

__global__ void gaussianHorizontalKernel(const float* input, float* temp, int width, int height, int channels, int kernelSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || c >= channels) {
        return;
    }

    const int radius = kernelSize / 2;
    float sum = 0.0f;
    const int channelOffset = c * width * height;

    for (int kx = -radius; kx <= radius; ++kx) {
        int nx = x + kx;
        // clamping boundaries
        if (nx < 0) nx = 0;
        if (nx >= width) nx = width - 1;

        float pixel = input[channelOffset + y * width + nx];
        float weight = c_kernel[kx + radius];
        sum += pixel * weight;
    }

    temp[channelOffset + y * width + x] = sum;
}

__global__ void gaussianVerticalKernel(const float* temp, float* output, int width, int height, int channels, int kernelSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || c >= channels) {
        return;
    }

    const int radius = kernelSize / 2;
    float sum = 0.0f;
    const int channelOffset = c * width * height;

    for (int ky = -radius; ky <= radius; ++ky) {
        int ny = y + ky;
        // clamping boundaries
        if (ny < 0) ny = 0;
        if (ny >= height) ny = height - 1;

        float pixel = temp[channelOffset + ny * width + x];
        float weight = c_kernel[ky + radius];
        sum += pixel * weight;
    }

    output[channelOffset + y * width + x] = sum;
}

namespace Filters {

void gaussianCUDA_NoAlloc(const float* d_input, float* d_temp, float* d_output, int width, int height, int channels, int kernelSize) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (channels + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    // Launch horizontal kernel
    gaussianHorizontalKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_temp, width, height, channels, kernelSize);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Horizontal kernel launch failed: ") + cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Horizontal kernel execution failed: ") + cudaGetErrorString(err));
    }

    // Launch vertical kernel
    gaussianVerticalKernel<<<numBlocks, threadsPerBlock>>>(d_temp, d_output, width, height, channels, kernelSize);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Vertical kernel launch failed: ") + cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Vertical kernel execution failed: ") + cudaGetErrorString(err));
    }
}

void gaussianCUDA_UploadKernel(const float* h_kernel, int kernelSize) {
    if (kernelSize > MAX_KERNEL_SIZE) {
        throw std::invalid_argument("Kernel size exceeds maximum constant memory array size.");
    }
    cudaMemcpyToSymbol(c_kernel, h_kernel, kernelSize * sizeof(float));
}

Image gaussianCUDA(const Image& input, int kernelSize, float sigma) {
    if (kernelSize <= 1 || sigma <= 0.0f) {
        return input;
    }
    if (kernelSize > MAX_KERNEL_SIZE) {
        throw std::invalid_argument("Kernel size exceeds maximum constant memory array size.");
    }

    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    Image output(width, height, channels);

    // computing 1D gaussian kernel on CPU
    const int radius = kernelSize / 2;
    std::vector<float> h_kernel(kernelSize);
    float kernelSum = 0.0f;
    const float twoSigmaSq = 2.0f * sigma * sigma;

    for (int i = 0; i < kernelSize; ++i) {
        float x = static_cast<float>(i - radius);
        h_kernel[i] = std::exp(-(x * x) / twoSigmaSq);
        kernelSum += h_kernel[i];
    }

    // normalizing kernel
    const float invSum = 1.0f / kernelSum;
    for (float& val : h_kernel) {
        val *= invSum;
    }

    const size_t imgSize = width * height * channels * sizeof(float);

    float* d_input = nullptr;
    float* d_temp = nullptr;
    float* d_output = nullptr;

    // allocating memory in GPU
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_temp, imgSize);
    cudaMalloc(&d_output, imgSize);

    // copying data from host to device
    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kernel, h_kernel.data(), kernelSize * sizeof(float));

    // launching kernels
    gaussianCUDA_NoAlloc(d_input, d_temp, d_output, width, height, channels, kernelSize);

    // copy the result back
    cudaMemcpy(output.getData(), d_output, imgSize, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);

    return output;
}

} // namespace Filters