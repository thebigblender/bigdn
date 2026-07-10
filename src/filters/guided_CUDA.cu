#include "filters.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define COARSEN_L 4
#define MAX_RADIUS 64

//box filter horizontal pass
__global__ void boxFilterHorizontalDualSharedKernel(
    const float* __restrict__ input1, const float* __restrict__ input2,
    float* __restrict__ temp1, float* __restrict__ temp2,
    const int width, const int height, const int channels,
    const int kernelSize, const float invKernelSize)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int blockStartRow = blockIdx.y * BLOCK_SIZE_Y;
    const int y = blockStartRow + ty;
    const int blockStartCol = blockIdx.x * (BLOCK_SIZE_X * COARSEN_L);
    const int c = blockIdx.z;
    const int radius = kernelSize / 2;
    const int channelOffset = c * width * height;

    __shared__ float s_data1[BLOCK_SIZE_Y][BLOCK_SIZE_X * COARSEN_L + 2 * MAX_RADIUS];
    __shared__ float s_data2[BLOCK_SIZE_Y][BLOCK_SIZE_X * COARSEN_L + 2 * MAX_RADIUS];

    const int sharedSize = BLOCK_SIZE_X * COARSEN_L + 2 * radius;

    if (y < height && c < channels) {
        const int rowOffset = y * width;
        for (int i = tx; i < sharedSize; i += BLOCK_SIZE_X) {
            const int globalX = blockStartCol - radius + i;
            const int clampedX = (globalX < 0) ? 0 : ((globalX >= width) ? width - 1 : globalX);
            const int idx = channelOffset + rowOffset + clampedX;
            s_data1[ty][i] = input1[idx];
            s_data2[ty][i] = input2[idx];
        }
    }

    __syncthreads();

    if (y < height && c < channels) {
        const int rowOffset = y * width;
        const int localStart = tx * COARSEN_L;

        float sum1 = 0.0f;
        float sum2 = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernelSize; ++k) {
            sum1 += s_data1[ty][localStart + k];
            sum2 += s_data2[ty][localStart + k];
        }
        int globalX = blockStartCol + localStart;
        if (globalX < width) {
            const int outIdx = channelOffset + rowOffset + globalX;
            temp1[outIdx] = sum1 * invKernelSize;
            temp2[outIdx] = sum2 * invKernelSize;
        }

        for (int l = 1; l < COARSEN_L; ++l) {
            sum1 += s_data1[ty][localStart + l - 1 + kernelSize] - s_data1[ty][localStart + l - 1];
            sum2 += s_data2[ty][localStart + l - 1 + kernelSize] - s_data2[ty][localStart + l - 1];
            globalX = blockStartCol + localStart + l;
            if (globalX < width) {
                const int outIdx = channelOffset + rowOffset + globalX;
                temp1[outIdx] = sum1 * invKernelSize;
                temp2[outIdx] = sum2 * invKernelSize;
            }
        }
    }
}

//box filter vertical pass
__global__ void boxFilterVerticalDualSharedKernel(
    const float* __restrict__ temp1, const float* __restrict__ temp2,
    float* __restrict__ output1, float* __restrict__ output2,
    const int width, const int height, const int channels,
    const int kernelSize, const float invKernelSize)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int blockStartRow = blockIdx.y * (BLOCK_SIZE_Y * COARSEN_L);
    const int c = blockIdx.z;
    const int radius = kernelSize / 2;
    const int channelOffset = c * width * height;

    __shared__ float s_data1[BLOCK_SIZE_Y * COARSEN_L + 2 * MAX_RADIUS][BLOCK_SIZE_X + 1];
    __shared__ float s_data2[BLOCK_SIZE_Y * COARSEN_L + 2 * MAX_RADIUS][BLOCK_SIZE_X + 1];

    const int rowsToLoad = BLOCK_SIZE_Y * COARSEN_L + 2 * radius;

    for (int r = ty; r < rowsToLoad; r += BLOCK_SIZE_Y) {
        const int globalY = blockStartRow - radius + r;
        const int clampedY = (globalY < 0) ? 0 : ((globalY >= height) ? height - 1 : globalY);

        float val1 = 0.0f;
        float val2 = 0.0f;
        if (x < width) {
            const int idx = channelOffset + clampedY * width + x;
            val1 = temp1[idx];
            val2 = temp2[idx];
        }
        s_data1[r][tx] = val1;
        s_data2[r][tx] = val2;
    }

    __syncthreads();

    if (x < width && c < channels) {
        const int localStart = ty * COARSEN_L;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernelSize; ++k) {
            sum1 += s_data1[localStart + k][tx];
            sum2 += s_data2[localStart + k][tx];
        }
        int globalY = blockStartRow + localStart;
        if (globalY < height) {
            const int outIdx = channelOffset + globalY * width + x;
            output1[outIdx] = sum1 * invKernelSize;
            output2[outIdx] = sum2 * invKernelSize;
        }

        for (int l = 1; l < COARSEN_L; ++l) {
            sum1 += s_data1[localStart + l - 1 + kernelSize][tx] - s_data1[localStart + l - 1][tx];
            sum2 += s_data2[localStart + l - 1 + kernelSize][tx] - s_data2[localStart + l - 1][tx];
            globalY = blockStartRow + localStart + l;
            if (globalY < height) {
                const int outIdx = channelOffset + globalY * width + x;
                output1[outIdx] = sum1 * invKernelSize;
                output2[outIdx] = sum2 * invKernelSize;
            }
        }
    }
}

//box filter vertical pass fused with final output
__global__ void boxFilterVerticalDualAndFinalKernel(
    const float* __restrict__ temp_a, const float* __restrict__ temp_b,
    const float* __restrict__ guidance, float* __restrict__ output,
    const int width, const int height, const int channels,
    const int kernelSize, const float invKernelSize)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int blockStartRow = blockIdx.y * (BLOCK_SIZE_Y * COARSEN_L);
    const int c = blockIdx.z;
    const int radius = kernelSize / 2;
    const int channelOffset = c * width * height;

    __shared__ float s_data1[BLOCK_SIZE_Y * COARSEN_L + 2 * MAX_RADIUS][BLOCK_SIZE_X + 1];
    __shared__ float s_data2[BLOCK_SIZE_Y * COARSEN_L + 2 * MAX_RADIUS][BLOCK_SIZE_X + 1];

    const int rowsToLoad = BLOCK_SIZE_Y * COARSEN_L + 2 * radius;

    for (int r = ty; r < rowsToLoad; r += BLOCK_SIZE_Y) {
        const int globalY = blockStartRow - radius + r;
        const int clampedY = (globalY < 0) ? 0 : ((globalY >= height) ? height - 1 : globalY);

        float val1 = 0.0f;
        float val2 = 0.0f;
        if (x < width) {
            const int idx = channelOffset + clampedY * width + x;
            val1 = temp_a[idx];
            val2 = temp_b[idx];
        }
        s_data1[r][tx] = val1;
        s_data2[r][tx] = val2;
    }

    __syncthreads();

    if (x < width && c < channels) {
        const int localStart = ty * COARSEN_L;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernelSize; ++k) {
            sum1 += s_data1[localStart + k][tx];
            sum2 += s_data2[localStart + k][tx];
        }
        int globalY = blockStartRow + localStart;
        if (globalY < height) {
            const int outIdx = channelOffset + globalY * width + x;
            float mean_a = sum1 * invKernelSize;
            float mean_b = sum2 * invKernelSize;
            output[outIdx] = mean_a * guidance[outIdx] + mean_b;
        }

        for (int l = 1; l < COARSEN_L; ++l) {
            sum1 += s_data1[localStart + l - 1 + kernelSize][tx] - s_data1[localStart + l - 1][tx];
            sum2 += s_data2[localStart + l - 1 + kernelSize][tx] - s_data2[localStart + l - 1][tx];
            globalY = blockStartRow + localStart + l;
            if (globalY < height) {
                const int outIdx = channelOffset + globalY * width + x;
                float mean_a = sum1 * invKernelSize;
                float mean_b = sum2 * invKernelSize;
                output[outIdx] = mean_a * guidance[outIdx] + mean_b;
            }
        }
    }
}

//helper to compute elementwise products
__global__ void computeIPandIIKernel(const float4* __restrict__ input, const float4* __restrict__ guidance, float4* __restrict__ I_p, float4* __restrict__ I_sq, const int size4) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        const float4 i_val = guidance[idx];
        const float4 p_val = input[idx];
        float4 sq_val, p_prod;

        sq_val.x = i_val.x * i_val.x;
        sq_val.y = i_val.y * i_val.y;
        sq_val.z = i_val.z * i_val.z;
        sq_val.w = i_val.w * i_val.w;

        p_prod.x = i_val.x * p_val.x;
        p_prod.y = i_val.y * p_val.y;
        p_prod.z = i_val.z * p_val.z;
        p_prod.w = i_val.w * p_val.w;

        I_sq[idx] = sq_val;
        I_p[idx] = p_prod;
    }
}

//helper to compute regression coeffs
__global__ void computeABKernel(
    const float4* __restrict__ mean_I, const float4* __restrict__ mean_p,
    const float4* __restrict__ mean_II, const float4* __restrict__ mean_Ip,
    float4* __restrict__ a, float4* __restrict__ b,
    const float eps, const int size4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        const float4 m_I = mean_I[idx];
        const float4 m_p = mean_p[idx];
        const float4 m_II = mean_II[idx];
        const float4 m_Ip = mean_Ip[idx];

        float4 a_val, b_val;

        float var_I_x = m_II.x - m_I.x * m_I.x;
        float cov_Ip_x = m_Ip.x - m_I.x * m_p.x;
        a_val.x = cov_Ip_x / (var_I_x + eps);
        b_val.x = m_p.x - a_val.x * m_I.x;

        float var_I_y = m_II.y - m_I.y * m_I.y;
        float cov_Ip_y = m_Ip.y - m_I.y * m_p.y;
        a_val.y = cov_Ip_y / (var_I_y + eps);
        b_val.y = m_p.y - a_val.y * m_I.y;

        float var_I_z = m_II.z - m_I.z * m_I.z;
        float cov_Ip_z = m_Ip.z - m_I.z * m_p.z;
        a_val.z = cov_Ip_z / (var_I_z + eps);
        b_val.z = m_p.z - a_val.z * m_I.z;

        float var_I_w = m_II.w - m_I.w * m_I.w;
        float cov_Ip_w = m_Ip.w - m_I.w * m_p.w;
        a_val.w = cov_Ip_w / (var_I_w + eps);
        b_val.w = m_p.w - a_val.w * m_I.w;

        a[idx] = a_val;
        b[idx] = b_val;
    }
}

namespace Filters {

//helper to launch horizontal and vertical box filter passes
static void boxFilterCUDADual(
    const float* d_input1, const float* d_input2,
    float* d_temp1, float* d_temp2,
    float* d_output1, float* d_output2,
    const int width, const int height, const int channels,
    const int kernelSize)
{
    const float invKernelSize = 1.0f / kernelSize;
    dim3 hThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 hGrid((width + BLOCK_SIZE_X * COARSEN_L - 1) / (BLOCK_SIZE_X * COARSEN_L), (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, channels);
    boxFilterHorizontalDualSharedKernel<<<hGrid, hThreads>>>(d_input1, d_input2, d_temp1, d_temp2, width, height, channels, kernelSize, invKernelSize);

    dim3 vThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 vGrid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y * COARSEN_L - 1) / (BLOCK_SIZE_Y * COARSEN_L), channels);
    boxFilterVerticalDualSharedKernel<<<vGrid, vThreads>>>(d_temp1, d_temp2, d_output1, d_output2, width, height, channels, kernelSize, invKernelSize);
}

void guidedCUDA_NoAlloc(const float* d_input, const float* d_guidance, float* d_output,
                        float* d_temp1, float* d_temp2, float* d_mean_I, float* d_mean_II, float* d_mean_Ip,
                        float* d_a, float* d_b,
                        int width, int height, int channels, int kernelSize, float eps) {
    const int totalPixels = width * height * channels;
    const int size4 = totalPixels / 4;
    const int blockSize = 256;
    const int gridSize = (size4 + blockSize - 1) / blockSize;
    const float invKernelSize = 1.0f / kernelSize;

    //local means
    boxFilterCUDADual(d_guidance, d_input, d_temp1, d_temp2, d_mean_I, d_output, width, height, channels, kernelSize);

    //local correlations
    computeIPandIIKernel<<<gridSize, blockSize>>>((const float4*)d_input, (const float4*)d_guidance, (float4*)d_b, (float4*)d_a, size4);

    boxFilterCUDADual(d_a, d_b, d_temp1, d_temp2, d_mean_II, d_mean_Ip, width, height, channels, kernelSize);

    //computing var, covar, and coeffs a and b
    computeABKernel<<<gridSize, blockSize>>>((const float4*)d_mean_I, (const float4*)d_output, (const float4*)d_mean_II, (const float4*)d_mean_Ip, (float4*)d_a, (float4*)d_b, eps, size4);

    //avging the coeffs over local window
    dim3 hThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 hGrid((width + BLOCK_SIZE_X * COARSEN_L - 1) / (BLOCK_SIZE_X * COARSEN_L), (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, channels);
    boxFilterHorizontalDualSharedKernel<<<hGrid, hThreads>>>(d_a, d_b, d_temp1, d_temp2, width, height, channels, kernelSize, invKernelSize);

    //final
    dim3 vThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 vGrid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y * COARSEN_L - 1) / (BLOCK_SIZE_Y * COARSEN_L), channels);
    boxFilterVerticalDualAndFinalKernel<<<vGrid, vThreads>>>(d_temp1, d_temp2, d_guidance, d_output, width, height, channels, kernelSize, invKernelSize);
}

Image guidedCUDA(const Image& input, const Image& guidance, int kernelSize, float eps) {
    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    const int totalPixels = width * height * channels;
    size_t imgSize = totalPixels * sizeof(float);

    Image output(width, height, channels);

    float* d_workspace = nullptr;
    cudaMalloc(&d_workspace, 10 * imgSize);

    float* d_input = d_workspace;
    float* d_guidance = d_workspace + totalPixels;
    float* d_output = d_workspace + 2 * totalPixels;
    float* d_temp1 = d_workspace + 3 * totalPixels;
    float* d_temp2 = d_workspace + 4 * totalPixels;
    float* d_mean_I = d_workspace + 5 * totalPixels;
    float* d_mean_II = d_workspace + 6 * totalPixels;
    float* d_mean_Ip = d_workspace + 7 * totalPixels;
    float* d_a = d_workspace + 8 * totalPixels;
    float* d_b = d_workspace + 9 * totalPixels;

    

    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_guidance, guidance.getData(), imgSize, cudaMemcpyHostToDevice);

    guidedCUDA_NoAlloc(d_input, d_guidance, d_output,
                       d_temp1, d_temp2, d_mean_I, d_mean_II, d_mean_Ip,
                       d_a, d_b,
                       width, height, channels, kernelSize, eps);

    cudaMemcpy(output.getData(), d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_workspace);

    return output;
}

}
