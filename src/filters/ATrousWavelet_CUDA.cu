#include "filters.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// 1D B-spline kernel weights
__constant__ float c_atrous_kernel[5] = {
    0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f
};

// Pre-process: 
// Demodulate input RGB by albedo RGB and store in temp.
// We do NOT modify normal or albedo maps in-place.
__global__ void atrousPrepareKernel(
    const float* __restrict__ input, const float* __restrict__ normal, const float* __restrict__ albedo,
    float* __restrict__ temp, const int width, const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int idx = y * width + x;
        const int plane = width * height;

        float r = input[idx];
        float g = input[plane + idx];
        float b = input[2 * plane + idx];

        float ar = albedo[idx];
        float ag = albedo[plane + idx];
        float ab = albedo[2 * plane + idx];

        // Stable demodulation in RGB space
        temp[idx] = r / (ar + 0.12f);
        temp[plane + idx] = g / (ag + 0.12f);
        temp[2 * plane + idx] = b / (ab + 0.12f);
    }
}

// Single pass of A-Trous Wavelet filter with adaptive sigmas & bilateral squared distances
__global__ void atrousFilterPassKernel(
    const float* __restrict__ input_img, const float* __restrict__ normal, const float* __restrict__ albedo,
    float* __restrict__ output_img,
    const int width, const int height, const int stepSize,
    const float baseSigmaColor, const float sigmaNormal, const float sigmaAlbedo)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int plane = width * height;
        const int centerIdx = y * width + x;

        // Center pixel values (demodulated shading)
        float center_r = input_img[centerIdx];
        float center_g = input_img[plane + centerIdx];
        float center_b = input_img[2 * plane + centerIdx];
        float center_lum = 0.299f * center_r + 0.587f * center_g + 0.114f * center_b;

        // Center normal (normalize on the fly to avoid in-place input mutation)
        float c_nx = 2.0f * normal[centerIdx] - 1.0f;
        float c_ny = 2.0f * normal[plane + centerIdx] - 1.0f;
        float c_nz = 2.0f * normal[2 * plane + centerIdx] - 1.0f;
        float c_inv_len = rsqrtf(c_nx * c_nx + c_ny * c_ny + c_nz * c_nz + 1e-6f);
        c_nx *= c_inv_len; c_ny *= c_inv_len; c_nz *= c_inv_len;

        // Center albedo (read to scale the color sigma adaptively)
        float ar = albedo[centerIdx];
        float ag = albedo[plane + centerIdx];
        float ab = albedo[2 * plane + centerIdx];
        float c_a_gray = 0.299f * ar + 0.587f * ag + 0.114f * ab;

        // Adaptive color sigma: scale up in dark albedo regions to handle higher noise amplification
        float sigmaColorY = baseSigmaColor / (c_a_gray + 0.08f);
        sigmaColorY = fminf(2.0f, sigmaColorY);

        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_weight = 0.0f;

        // 5x5 B-spline convolution
        #pragma unroll
        for (int dy = -2; dy <= 2; ++dy) {
            int qy = y + dy * stepSize;
            qy = (qy < 0) ? 0 : ((qy >= height) ? height - 1 : qy);
            float h_y = c_atrous_kernel[dy + 2];

            #pragma unroll
            for (int dx = -2; dx <= 2; ++dx) {
                int qx = x + dx * stepSize;
                qx = (qx < 0) ? 0 : ((qx >= width) ? width - 1 : qx);
                float h_x = c_atrous_kernel[dx + 2];

                float kernel_weight = h_y * h_x;

                const int qIdx = qy * width + qx;

                // Sample shading values
                float q_r = input_img[qIdx];
                float q_g = input_img[plane + qIdx];
                float q_b = input_img[2 * plane + qIdx];
                float q_lum = 0.299f * q_r + 0.587f * q_g + 0.114f * q_b;

                // Normal difference (normalize on the fly)
                float q_nx = 2.0f * normal[qIdx] - 1.0f;
                float q_ny = 2.0f * normal[plane + qIdx] - 1.0f;
                float q_nz = 2.0f * normal[2 * plane + qIdx] - 1.0f;
                float q_inv_len = rsqrtf(q_nx * q_nx + q_ny * q_ny + q_nz * q_nz + 1e-6f);
                q_nx *= q_inv_len; q_ny *= q_inv_len; q_nz *= q_inv_len;

                float dot_n = c_nx * q_nx + c_ny * q_ny + c_nz * q_nz;
                float dist_normal = fmaxf(0.0f, 1.0f - dot_n);

                // Luminance difference (of demodulated shading)
                float dlum = center_lum - q_lum;
                float dist_lum_sq = dlum * dlum;

                // Edge weights (Bilateral formulation)
                float w_n = __expf(-dist_normal / (sigmaNormal * sigmaNormal + 1e-6f));
                float w_cy = __expf(-dist_lum_sq / (2.0f * sigmaColorY * sigmaColorY + 1e-6f));

                float weight = w_n * w_cy * kernel_weight;

                sum_r += weight * q_r;
                sum_g += weight * q_g;
                sum_b += weight * q_b;
                sum_weight += weight;
            }
        }

        output_img[centerIdx] = sum_r / (sum_weight + 1e-6f);
        output_img[plane + centerIdx] = sum_g / (sum_weight + 1e-6f);
        output_img[2 * plane + centerIdx] = sum_b / (sum_weight + 1e-6f);
    }
}

// Reconstruct: convert back to RGB, modulate back, and clamp to [0, 1]
__global__ void atrousReconstructKernel(
    const float* __restrict__ shading, const float* __restrict__ albedo, float* __restrict__ output,
    const int width, const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int idx = y * width + x;
        const int plane = width * height;

        float sr = shading[idx];
        float sg = shading[plane + idx];
        float sb = shading[2 * plane + idx];

        float ar = albedo[idx];
        float ag = albedo[plane + idx];
        float ab = albedo[2 * plane + idx];

        // Modulate back (shading * albedo)
        float r = sr * (ar + 0.12f);
        float g = sg * (ag + 0.12f);
        float b = sb * (ab + 0.12f);

        // Clamp final output to [0.0, 1.0]
        output[idx] = fmaxf(0.0f, fminf(1.0f, r));
        output[plane + idx] = fmaxf(0.0f, fminf(1.0f, g));
        output[2 * plane + idx] = fmaxf(0.0f, fminf(1.0f, b));
    }
}

namespace Filters {

void aTrousWaveletCUDA_NoAlloc(const float* d_input, const float* d_normal, const float* d_albedo, float* d_output,
                              float* d_temp1, float* d_temp2,
                              int width, int height, int channels, int passes,
                              float sigmaColor, float sigmaNormal, float sigmaAlbedo)
{
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    (void)channels;

    // 1. Preprocess: demodulate RGB, store in d_temp1
    atrousPrepareKernel<<<grid, threads>>>(d_input, d_normal, d_albedo, d_temp1, width, height);

    // 2. Ping-pong A-Trous Wavelet filter passes
    float* d_in_pass = d_temp1;
    float* d_out_pass = d_temp2;

    float currentSigmaColor = sigmaColor;

    for (int i = 0; i < passes; ++i) {
        int stepSize = 1 << i; // 1, 2, 4, 8, 16...
        atrousFilterPassKernel<<<grid, threads>>>(d_in_pass, d_normal, d_albedo, d_out_pass, width, height, stepSize, currentSigmaColor, sigmaNormal, sigmaAlbedo);
        
        // Color guidance gets weaker (larger sigma) every iteration to allow more averaging
        currentSigmaColor *= 1.1f;
        
        // Swap buffers for the next pass
        std::swap(d_in_pass, d_out_pass);
    }

    // 3. Reconstruct: modulate back, clamp, and write to d_output
    atrousReconstructKernel<<<grid, threads>>>(d_in_pass, d_albedo, d_output, width, height);
}


Image aTrousWaveletCUDA(const Image& input, const Image& normal, const Image& albedo, int passes, float sigmaColor, float sigmaNormal, float sigmaAlbedo) {
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
    float* d_temp1 = nullptr;
    float* d_temp2 = nullptr;

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_normal, imgSize);
    cudaMalloc(&d_albedo, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_temp1, imgSize);
    cudaMalloc(&d_temp2, imgSize);

    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, normal.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo, albedo.getData(), imgSize, cudaMemcpyHostToDevice);

    aTrousWaveletCUDA_NoAlloc(d_input, d_normal, d_albedo, d_output, d_temp1, d_temp2,
                              width, height, channels, passes, sigmaColor, sigmaNormal, sigmaAlbedo);

    cudaMemcpy(output.getData(), d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_normal);
    cudaFree(d_albedo);
    cudaFree(d_output);
    cudaFree(d_temp1);
    cudaFree(d_temp2);

    return output;
}

} // namespace Filters
