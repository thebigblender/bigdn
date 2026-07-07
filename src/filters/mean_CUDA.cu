#include "filters.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void meanFilterHorizontalKernel(const float* input, float* temp, int width, int height, int channels, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    //out of bounds check
    if (x >= width || y >= height || c >= channels) {
        return;
    }

    const int radius = kernelSize / 2;
    float sum = 0.0f;
    const int channelOffset = c * width * height;

    for (int kx = -radius; kx <= radius; ++kx) {
        //clamp2
        int nx = x + kx;
        if (nx < 0) nx = 0;
        if (nx >= width) nx = width - 1;

        sum += input[channelOffset + y * width + nx]; //sum of the whole thing which when div by ks gives avg
    }

    //writing avg to the corresponding pixel in the intermediate temp img
    temp[channelOffset + y * width + x] = sum / kernelSize;
}

__global__ void meanFilterVerticalKernel(const float* temp, float* output, int width, int height, int channels, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    //out of bounds check
    if (x >= width || y >= height || c >= channels) {
        return;
    }

    const int radius = kernelSize / 2;
    float sum = 0.0f;
    const int channelOffset = c * width * height;

    for (int ky = -radius; ky <= radius; ++ky) {
        //clamp1
        int ny = y + ky;
        if (ny < 0) ny = 0;
        if (ny >= height) ny = height - 1;

        sum += temp[channelOffset + ny * width + x]; //sum of the whole thing which when div by ks gives avg
    }

    //writing avg to the corresponding pixel in the output img
    output[channelOffset + y * width + x] = sum / kernelSize;
}

namespace Filters {

void meanCUDA_NoAlloc(const float* d_input, float* d_temp, float* d_output, int width, int height, int channels, int kernelSize) {
    //each block contains 16×16 threads -> 256 total. grid's z dim has one layer per channel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x, //120 for a 1080p img
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y, //68 for a 1080p img
        (channels + threadsPerBlock.z - 1) / threadsPerBlock.z //this will be 3
    );

    meanFilterHorizontalKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_temp, width, height, channels, kernelSize);
    meanFilterVerticalKernel<<<numBlocks, threadsPerBlock>>>(d_temp, d_output, width, height, channels, kernelSize);
}

Image meanCUDA(const Image& input, int kernelSize) {
    //le thing
    if (kernelSize <= 1) {
        return input;
    }

    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    Image output(width, height, channels);

    const size_t size = width * height * channels * sizeof(float); //num of bytes to store img on the GPU
    float* d_input = nullptr;
    float* d_temp = nullptr;
    float* d_output = nullptr;

    cudaError_t err;

    //allocating GPU mem for i/p img
    //d_input ptr is on the CPU, it just points towards an address in the GPU's mem space
    err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed for input: ") + cudaGetErrorString(err));
    }

    //allocating GPU mem for intermediate temp img
    err = cudaMalloc(&d_temp, size);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        throw std::runtime_error(std::string("cudaMalloc failed for temp: ") + cudaGetErrorString(err));
    }

    //allocating GPU mem for o/p img
    err = cudaMalloc(&d_output, size);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_temp);
        throw std::runtime_error(std::string("cudaMalloc failed for output: ") + cudaGetErrorString(err));
    }

    //copies input data in CPU into the space allocated in the GPU (d_input)
    err = cudaMemcpy(d_input, input.getData(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_temp);
        cudaFree(d_output);
        throw std::runtime_error(std::string("cudaMemcpy HostToDevice failed: ") + cudaGetErrorString(err));
    }

    //each block contains 16×16 threads -> 256 total. grid's z dim has one layer per channel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x, //120 for a 1080p img
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y, //68 for a 1080p img
        (channels + threadsPerBlock.z - 1) / threadsPerBlock.z //this will be 3
    );

    //launching the horizontal 1D CUDA kernel on the GPU
    meanFilterHorizontalKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_temp, width, height, channels, kernelSize);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_temp);
        cudaFree(d_output);
        throw std::runtime_error(std::string("Horizontal kernel launch failed: ") + cudaGetErrorString(err));
    }

    //launching the vertical 1D CUDA kernel on the GPU
    meanFilterVerticalKernel<<<numBlocks, threadsPerBlock>>>(d_temp, d_output, width, height, channels, kernelSize);

    //check for launch errors. checks if the kernel launch was successful
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_temp);
        cudaFree(d_output);
        throw std::runtime_error(std::string("Vertical kernel launch failed: ") + cudaGetErrorString(err));
    }

    //returns after the kernel has finished executing and reports runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_temp);
        cudaFree(d_output);
        throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
    }

    //copying the output data back to host memory
    err = cudaMemcpy(output.getData(), d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_temp);
        cudaFree(d_output);
        throw std::runtime_error(std::string("cudaMemcpy DeviceToHost failed: ") + cudaGetErrorString(err));
    }

    //GPU memory cleanup
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);

    return output;
}

} // namespace Filters
