#include "filters.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// 1d b-spline kernel weights for À-Trous Wavelet filter
__constant__ float c_vg_atrous_kernel[5] = {
    0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f
};

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

// Preprocess: demodulate input rgb by albedo rgb and store in temp
__global__ void vgAtrousPrepareKernel(
    const float* __restrict__ input, const float* __restrict__ albedo,
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

        // Stable demodulation in rgb space
        temp[idx] = r / (ar + 0.12f);
        temp[plane + idx] = g / (ag + 0.12f);
        temp[2 * plane + idx] = b / (ab + 0.12f);
    }
}

// Variance Estimation: Computes the local variance of luminance from the demodulated shading
__global__ void estimateVarianceKernel(
    const float* __restrict__ demodulated_img,
    float* __restrict__ variance_map,
    const int width, const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int plane = width * height;
        const int idx = y * width + x;

        float sum_L = 0.0f;
        float sum_L2 = 0.0f;
        float weight_sum = 0.0f;

        // Estimate variance in a 5x5 window for stability
        #pragma unroll
        for (int dy = -2; dy <= 2; ++dy) {
            int qy = y + dy;
            qy = (qy < 0) ? 0 : ((qy >= height) ? height - 1 : qy);
            #pragma unroll
            for (int dx = -2; dx <= 2; ++dx) {
                int qx = x + dx;
                qx = (qx < 0) ? 0 : ((qx >= width) ? width - 1 : qx);

                int qIdx = qy * width + qx;
                float r = demodulated_img[qIdx];
                float g = demodulated_img[plane + qIdx];
                float b = demodulated_img[2 * plane + qIdx];
                float L = 0.299f * r + 0.587f * g + 0.114f * b;

                sum_L += L;
                sum_L2 += L * L;
                weight_sum += 1.0f;
            }
        }

        float mean_L = sum_L / weight_sum;
        float mean_L2 = sum_L2 / weight_sum;
        float var = fmaxf(0.0f, mean_L2 - mean_L * mean_L);

        variance_map[idx] = var;
    }
}

// Box filter: smooths the variance map in a 3x3 window to avoid noise in the weights
__global__ void filterVarianceKernel(
    const float* __restrict__ raw_variance,
    float* __restrict__ filtered_variance,
    const int width, const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int idx = y * width + x;

        float sum_var = 0.0f;
        float weight_sum = 0.0f;

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            int qy = y + dy;
            qy = (qy < 0) ? 0 : ((qy >= height) ? height - 1 : qy);
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                int qx = x + dx;
                qx = (qx < 0) ? 0 : ((qx >= width) ? width - 1 : qx);

                sum_var += raw_variance[qy * width + qx];
                weight_sum += 1.0f;
            }
        }

        filtered_variance[idx] = sum_var / weight_sum;
    }
}

// Single pass of variance-guided À-Trous wavelet filter
__global__ void varianceGuidedAtrousFilterPassKernel(
    const float* __restrict__ input_img, const float* __restrict__ normal, const float* __restrict__ albedo,
    const float* __restrict__ variance_img,
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

        // Center normal normalized on the fly
        float c_nx = 2.0f * normal[centerIdx] - 1.0f;
        float c_ny = 2.0f * normal[plane + centerIdx] - 1.0f;
        float c_nz = 2.0f * normal[2 * plane + centerIdx] - 1.0f;
        float c_inv_len = rsqrtf(c_nx * c_nx + c_ny * c_ny + c_nz * c_nz + 1e-6f);
        c_nx *= c_inv_len; c_ny *= c_inv_len; c_nz *= c_inv_len;

        // Center albedo read for secondary guidance
        float ar = albedo[centerIdx];
        float ag = albedo[plane + centerIdx];
        float ab = albedo[2 * plane + centerIdx];
        float c_a_gray = 0.299f * ar + 0.587f * ag + 0.114f * ab;

        // Read standard deviation of local noise variance
        float center_var = variance_img[centerIdx];
        float std_dev = sqrtf(center_var);

        // Dynamically scale color sigma based on local noise standard deviation
        // We include a small baseline offset of 0.015f to preserve clean details in low-noise regions
        float sigmaColorY = baseSigmaColor * (std_dev + 0.015f);
        
        // Boost color sigma slightly in dark albedo regions where SNR is worse
        sigmaColorY /= (c_a_gray + 0.08f);
        sigmaColorY = fminf(2.0f, sigmaColorY);

        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_weight = 0.0f;

        // 5x5 b-spline convolution
        #pragma unroll
        for (int dy = -2; dy <= 2; ++dy) {
            int qy = y + dy * stepSize;
            qy = (qy < 0) ? 0 : ((qy >= height) ? height - 1 : qy);
            float h_y = c_vg_atrous_kernel[dy + 2];

            #pragma unroll
            for (int dx = -2; dx <= 2; ++dx) {
                int qx = x + dx * stepSize;
                qx = (qx < 0) ? 0 : ((qx >= width) ? width - 1 : qx);
                float h_x = c_vg_atrous_kernel[dx + 2];

                float kernel_weight = h_y * h_x;

                const int qIdx = qy * width + qx;

                // Sample shading values
                float q_r = input_img[qIdx];
                float q_g = input_img[plane + qIdx];
                float q_b = input_img[2 * plane + qIdx];
                float q_lum = 0.299f * q_r + 0.587f * q_g + 0.114f * q_b;

                // Normal difference normalized on the fly
                float q_nx = 2.0f * normal[qIdx] - 1.0f;
                float q_ny = 2.0f * normal[plane + qIdx] - 1.0f;
                float q_nz = 2.0f * normal[2 * plane + qIdx] - 1.0f;
                float q_inv_len = rsqrtf(q_nx * q_nx + q_ny * q_ny + q_nz * q_nz + 1e-6f);
                q_nx *= q_inv_len; q_ny *= q_inv_len; q_nz *= q_inv_len;

                float dot_n = c_nx * q_nx + c_ny * q_ny + c_nz * q_nz;
                float dist_normal = fmaxf(0.0f, 1.0f - dot_n);

                // Luminance difference of shading
                float dlum = center_lum - q_lum;
                float dist_lum_sq = dlum * dlum;

                // Bilateral weights
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

// Reconstruct: convert back to rgb, modulate back, and clamp
__global__ void vgAtrousReconstructKernel(
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

        // Modulate back
        float r = sr * (ar + 0.12f);
        float g = sg * (ag + 0.12f);
        float b = sb * (ab + 0.12f);

        // Clamp output to [0, 1]
        output[idx] = fmaxf(0.0f, fminf(1.0f, r));
        output[plane + idx] = fmaxf(0.0f, fminf(1.0f, g));
        output[2 * plane + idx] = fmaxf(0.0f, fminf(1.0f, b));
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
    size_t singleChannelSize = width * height * sizeof(float);

    Image output(width, height, channels);

    float* d_input = nullptr;
    float* d_normal = nullptr;
    float* d_albedo = nullptr;
    float* d_output = nullptr;
    float* d_temp_median = nullptr;
    float* d_variance_raw = nullptr;
    float* d_variance_filtered = nullptr;
    float* d_temp1 = nullptr;
    float* d_temp2 = nullptr;

    // Allocate GPU memory
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_normal, imgSize);
    cudaMalloc(&d_albedo, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_temp_median, imgSize);
    cudaMalloc(&d_variance_raw, singleChannelSize);
    cudaMalloc(&d_variance_filtered, singleChannelSize);
    cudaMalloc(&d_temp1, imgSize);
    cudaMalloc(&d_temp2, imgSize);

    // Copy input data from host to device
    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, normal.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo, albedo.getData(), imgSize, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // 1. Run masked median pre-filtering pass to eliminate salt-and-pepper outliers
    int mK = medianKernelSize;
    if (mK != 3 && mK != 5) {
        mK = (mK < 4) ? 3 : 5;
    }
    maskedMedianFilterKernel<<<grid, threads>>>(d_input, d_temp_median, width, height, mK, medianThreshold);

    // 2. Preprocess: Demodulate outlier-corrected beauty by albedo
    vgAtrousPrepareKernel<<<grid, threads>>>(d_temp_median, d_albedo, d_temp1, width, height);

    // 3. Estimate local variance from demodulated shading
    estimateVarianceKernel<<<grid, threads>>>(d_temp1, d_variance_raw, width, height);

    // 4. Smooth out the estimated variance to avoid noise-in-weights
    filterVarianceKernel<<<grid, threads>>>(d_variance_raw, d_variance_filtered, width, height);

    // 5. Run ping-pong À-Trous passes with variance-guided bilateral weights
    float* d_in_pass = d_temp1;
    float* d_out_pass = d_temp2;
    float currentSigmaColor = sigmaColor;

    for (int i = 0; i < passes; ++i) {
        int stepSize = 1 << i;
        varianceGuidedAtrousFilterPassKernel<<<grid, threads>>>(
            d_in_pass, d_normal, d_albedo, d_variance_filtered, d_out_pass,
            width, height, stepSize, currentSigmaColor, sigmaNormal, sigmaAlbedo
        );

        // Relax color guidance parameter slightly each pass
        currentSigmaColor *= 1.1f;
        std::swap(d_in_pass, d_out_pass);
    }

    // 6. Reconstruct: modulate back by albedo and clamp
    vgAtrousReconstructKernel<<<grid, threads>>>(d_in_pass, d_albedo, d_output, width, height);

    // Copy result back to host
    cudaMemcpy(output.getData(), d_output, imgSize, cudaMemcpyDeviceToHost);

    // Cleanup GPU resources
    cudaFree(d_input);
    cudaFree(d_normal);
    cudaFree(d_albedo);
    cudaFree(d_output);
    cudaFree(d_temp_median);
    cudaFree(d_variance_raw);
    cudaFree(d_variance_filtered);
    cudaFree(d_temp1);
    cudaFree(d_temp2);

    return output;
}

} // namespace Filters
