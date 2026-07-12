#include "filters.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define COARSEN_L 4
#define MAX_RADIUS 64

// constant memory for ycbcr conversion matrices
__constant__ float c_rgb2yuv[9] = {
     0.299000f,  0.587000f,  0.114000f,
    -0.168736f, -0.331264f,  0.500000f,
     0.500000f, -0.418688f, -0.081312f
};
__constant__ float c_yuv2rgb[4] = {
    1.402000f, -0.344136f, -0.714136f, 1.772000f
};

// preprocess and correlation fusion (vectorized float4)
__global__ void jointPrepareKernel(
    const float4* __restrict__ input, const float4* __restrict__ normal, const float4* __restrict__ albedo,
    float4* __restrict__ shading, float4* __restrict__ combined_guidance,
    float4* __restrict__ d_a, float4* __restrict__ d_b,
    const int size4, const int plane4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        // load beauty channels
        float4 r_v = input[idx];
        float4 g_v = input[idx + plane4];
        float4 b_v = input[idx + 2 * plane4];

        // load normal channels
        float4 nx_v = normal[idx];
        float4 ny_v = normal[idx + plane4];
        float4 nz_v = normal[idx + 2 * plane4];

        // load albedo channels
        float4 ar_v = albedo[idx];
        float4 ag_v = albedo[idx + plane4];
        float4 ab_v = albedo[idx + 2 * plane4];

        float4 g_val;
        float4 shading_y;
        float4 cb_val;
        float4 cr_val;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float r = (&r_v.x)[i];
            float g = (&g_v.x)[i];
            float b = (&b_v.x)[i];

            float nx = 2.0f * (&nx_v.x)[i] - 1.0f;
            float ny = 2.0f * (&ny_v.x)[i] - 1.0f;
            float nz = 2.0f * (&nz_v.x)[i] - 1.0f;
            
            float inv_len = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-6f);
            nx *= inv_len;
            ny *= inv_len;
            nz *= inv_len;

            float n_gray = 0.299f * fabsf(nx) + 0.587f * fabsf(ny) + 0.114f * fabsf(nz);

            float ar = (&ar_v.x)[i];
            float ag = (&ag_v.x)[i];
            float ab = (&ab_v.x)[i];
            float a_gray = c_rgb2yuv[0] * ar + c_rgb2yuv[1] * ag + c_rgb2yuv[2] * ab;

            // combined guidance
            float g_px = 2.0f * n_gray + 0.5f * a_gray;
            (&g_val.x)[i] = g_px;

            float y_val  =  c_rgb2yuv[0] * r + c_rgb2yuv[1] * g + c_rgb2yuv[2] * b;
            (&cb_val.x)[i] = c_rgb2yuv[3] * r + c_rgb2yuv[4] * g + c_rgb2yuv[5] * b + 0.5f;
            (&cr_val.x)[i] = c_rgb2yuv[6] * r + c_rgb2yuv[7] * g + c_rgb2yuv[8] * b + 0.5f;

            // demodulate shading by albedo
            (&shading_y.x)[i] = y_val / (a_gray + 0.15f);
        }

        // store combined guidance
        combined_guidance[idx] = g_val;
        combined_guidance[idx + plane4] = g_val;
        combined_guidance[idx + 2 * plane4] = g_val;

        // store shading
        shading[idx] = shading_y;
        shading[idx + plane4] = cb_val;
        shading[idx + 2 * plane4] = cr_val;

        // precompute g^2
        float4 g2;
        g2.x = g_val.x * g_val.x;
        g2.y = g_val.y * g_val.y;
        g2.z = g_val.z * g_val.z;
        g2.w = g_val.w * g_val.w;

        d_a[idx] = g2;
        d_a[idx + plane4] = g2;
        d_a[idx + 2 * plane4] = g2;

        // precompute g * shading
        float4 gp_y, gp_cb, gp_cr;
        gp_y.x = g_val.x * shading_y.x;
        gp_y.y = g_val.y * shading_y.y;
        gp_y.z = g_val.z * shading_y.z;
        gp_y.w = g_val.w * shading_y.w;

        gp_cb.x = g_val.x * cb_val.x;
        gp_cb.y = g_val.y * cb_val.y;
        gp_cb.z = g_val.z * cb_val.z;
        gp_cb.w = g_val.w * cb_val.w;

        gp_cr.x = g_val.x * cr_val.x;
        gp_cr.y = g_val.y * cr_val.y;
        gp_cr.z = g_val.z * cr_val.z;
        gp_cr.w = g_val.w * cr_val.w;

        d_b[idx] = gp_y;
        d_b[idx + plane4] = gp_cb;
        d_b[idx + 2 * plane4] = gp_cr;
    }
}

// reconstruct and convert back to rgb (vectorized float4)
__global__ void jointReconstructKernel(
    const float4* __restrict__ shading, const float4* __restrict__ albedo, float4* __restrict__ output,
    const int size4, const int plane4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 shading_y = shading[idx];
        float4 cb = shading[idx + plane4];
        float4 cr = shading[idx + 2 * plane4];

        float4 ar_v = albedo[idx];
        float4 ag_v = albedo[idx + plane4];
        float4 ab_v = albedo[idx + 2 * plane4];

        float4 r_out, g_out, b_out;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float sy = (&shading_y.x)[i];
            float cb_val = (&cb.x)[i];
            float cr_val = (&cr.x)[i];

            float ar = (&ar_v.x)[i];
            float ag = (&ag_v.x)[i];
            float ab = (&ab_v.x)[i];

            float a_gray = c_rgb2yuv[0] * ar + c_rgb2yuv[1] * ag + c_rgb2yuv[2] * ab;

            // modulate back
            float y_val = sy * (a_gray + 0.15f);

            // ycbcr to rgb
            float r = y_val + c_yuv2rgb[0] * (cr_val - 0.5f);
            float g = y_val + c_yuv2rgb[1] * (cb_val - 0.5f) + c_yuv2rgb[2] * (cr_val - 0.5f);
            float b = y_val + c_yuv2rgb[3] * (cb_val - 0.5f);

            // clamp output to [0, 1]
            (&r_out.x)[i] = fmaxf(0.0f, fminf(1.0f, r));
            (&g_out.x)[i] = fmaxf(0.0f, fminf(1.0f, g));
            (&b_out.x)[i] = fmaxf(0.0f, fminf(1.0f, b));
        }

        output[idx] = r_out;
        output[idx + plane4] = g_out;
        output[idx + 2 * plane4] = b_out;
    }
}

// box filter horizontal pass
__global__ void jointBoxFilterHorizontalDualSharedKernel(
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

// box filter vertical pass
__global__ void jointBoxFilterVerticalDualSharedKernel(
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

// box filter vertical pass fused with final output
__global__ void jointBoxFilterVerticalDualAndFinalKernel(
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

// compute regression coeffs with adaptive epsilon
__global__ void jointComputeABKernel(
    const float4* __restrict__ mean_I, const float4* __restrict__ mean_p,
    const float4* __restrict__ mean_II, const float4* mean_Ip,
    float4* a, float4* __restrict__ b,
    const int size4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        const float4 m_I = mean_I[idx];
        const float4 m_p = mean_p[idx];
        const float4 m_II = mean_II[idx];
        const float4 m_Ip = mean_Ip[idx];

        float4 a_val, b_val;

        // channel x
        float var_I_x = m_II.x - m_I.x * m_I.x;
        float cov_Ip_x = m_Ip.x - m_I.x * m_p.x;
        float eps_x = 0.01f * var_I_x + 1e-4f;
        a_val.x = cov_Ip_x / (var_I_x + eps_x);
        b_val.x = m_p.x - a_val.x * m_I.x;

        // channel y
        float var_I_y = m_II.y - m_I.y * m_I.y;
        float cov_Ip_y = m_Ip.y - m_I.y * m_p.y;
        float eps_y = 0.01f * var_I_y + 1e-4f;
        a_val.y = cov_Ip_y / (var_I_y + eps_y);
        b_val.y = m_p.y - a_val.y * m_I.y;

        // channel z
        float var_I_z = m_II.z - m_I.z * m_I.z;
        float cov_Ip_z = m_Ip.z - m_I.z * m_p.z;
        float eps_z = 0.01f * var_I_z + 1e-4f;
        a_val.z = cov_Ip_z / (var_I_z + eps_z);
        b_val.z = m_p.z - a_val.z * m_I.z;

        // channel w
        float var_I_w = m_II.w - m_I.w * m_I.w;
        float cov_Ip_w = m_Ip.w - m_I.w * m_p.w;
        float eps_w = 0.01f * var_I_w + 1e-4f;
        a_val.w = cov_Ip_w / (var_I_w + eps_w);
        b_val.w = m_p.w - a_val.w * m_I.w;

        a[idx] = a_val;
        b[idx] = b_val;
    }
}

namespace Filters {

// helper to launch horizontal and vertical box filter passes
static void jointBoxFilterCUDADual(
    const float* d_input1, const float* d_input2,
    float* d_temp1, float* d_temp2,
    float* d_output1, float* d_output2,
    const int width, const int height, const int channels,
    const int kernelSize)
{
    const float invKernelSize = 1.0f / kernelSize;
    dim3 hThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 hGrid((width + BLOCK_SIZE_X * COARSEN_L - 1) / (BLOCK_SIZE_X * COARSEN_L), (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, channels);
    jointBoxFilterHorizontalDualSharedKernel<<<hGrid, hThreads>>>(d_input1, d_input2, d_temp1, d_temp2, width, height, channels, kernelSize, invKernelSize);

    dim3 vThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 vGrid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y * COARSEN_L - 1) / (BLOCK_SIZE_Y * COARSEN_L), channels);
    jointBoxFilterVerticalDualSharedKernel<<<vGrid, vThreads>>>(d_temp1, d_temp2, d_output1, d_output2, width, height, channels, kernelSize, invKernelSize);
}

void jointGuidedCUDA_NoAlloc(const float* d_input, const float* d_normal, const float* d_albedo, float* d_output,
                            float* d_shading, float* d_temp1, float* d_temp2, float* d_mean_I, float* d_mean_II, float* d_mean_Ip,
                            float* d_a, float* d_b,
                            int width, int height, int channels, int kernelSize, float eps) 
{
    const int totalPixels = width * height * channels;
    const int size4 = totalPixels / 4;
    const int blockSize = 256;
    const int gridSize = (size4 + blockSize - 1) / blockSize;
    const float invKernelSize = 1.0f / kernelSize;
    (void)eps;

    // prepare and correlation fusion
    const int plane = width * height;
    const int plane4 = plane / 4;
    const int planeGridSize = (plane4 + blockSize - 1) / blockSize;
    
    // write guidance directly to d_mean_I
    jointPrepareKernel<<<planeGridSize, blockSize>>>(
        (const float4*)d_input, (const float4*)d_normal, (const float4*)d_albedo,
        (float4*)d_shading, (float4*)d_mean_I,
        (float4*)d_a, (float4*)d_b,
        plane4 * 3, plane4
    );

    // local means
    // outputs go to mean_II and output
    jointBoxFilterCUDADual(d_mean_I, d_shading, d_temp1, d_temp2, d_mean_II, d_output, width, height, channels, kernelSize);

    // box filter precomputed products
    // outputs go to mean_Ip and d_a
    jointBoxFilterCUDADual(d_a, d_b, d_temp1, d_temp2, d_mean_Ip, d_a, width, height, channels, kernelSize);

    // compute regression coefficients
    // pass intermediate means
    // outputs go to d_a and d_b
    jointComputeABKernel<<<gridSize, blockSize>>>((const float4*)d_mean_II, (const float4*)d_output, (const float4*)d_mean_Ip, (const float4*)d_a, (float4*)d_a, (float4*)d_b, size4);

    // average coefficients over local windows
    dim3 hThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 hGrid((width + BLOCK_SIZE_X * COARSEN_L - 1) / (BLOCK_SIZE_X * COARSEN_L), (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, channels);
    jointBoxFilterHorizontalDualSharedKernel<<<hGrid, hThreads>>>(d_a, d_b, d_temp1, d_temp2, width, height, channels, kernelSize, invKernelSize);

    // vertical pass and compute filtered shading
    dim3 vThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 vGrid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y * COARSEN_L - 1) / (BLOCK_SIZE_Y * COARSEN_L), channels);
    jointBoxFilterVerticalDualAndFinalKernel<<<vGrid, vThreads>>>(d_temp1, d_temp2, d_mean_I, d_shading, width, height, channels, kernelSize, invKernelSize);

    // reconstruct: convert to rgb, modulate, clamp, and store
    jointReconstructKernel<<<planeGridSize, blockSize>>>(
        (const float4*)d_shading, (const float4*)d_albedo, (float4*)d_output,
        plane4, plane4
    );
}

Image jointGuidedCUDA(const Image& input, const Image& normal, const Image& albedo, int kernelSize, float eps) {
    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    const int totalPixels = width * height * channels;
    size_t imgSize = totalPixels * sizeof(float);

    Image output(width, height, channels);

    float* d_workspace = nullptr;
    // allocate memory in GPU
    cudaMalloc(&d_workspace, 15 * imgSize);

    float* d_input   = d_workspace;
    float* d_normal  = d_workspace + totalPixels;
    float* d_albedo  = d_workspace + 2 * totalPixels;
    float* d_output  = d_workspace + 3 * totalPixels;
    float* d_shading = d_workspace + 4 * totalPixels;
    float* d_temp1   = d_workspace + 5 * totalPixels;
    float* d_temp2   = d_workspace + 6 * totalPixels;
    float* d_mean_I  = d_workspace + 7 * totalPixels;
    float* d_mean_II = d_workspace + 8 * totalPixels;
    float* d_mean_Ip = d_workspace + 9 * totalPixels;
    float* d_a       = d_workspace + 10 * totalPixels;
    float* d_b       = d_workspace + 11 * totalPixels;

    // copy data from host to device
    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, normal.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo, albedo.getData(), imgSize, cudaMemcpyHostToDevice);

    jointGuidedCUDA_NoAlloc(d_input, d_normal, d_albedo, d_output,
                            d_shading, d_temp1, d_temp2, d_mean_I, d_mean_II, d_mean_Ip,
                            d_a, d_b,
                            width, height, channels, kernelSize, eps);

    // copy the result back
    cudaMemcpy(output.getData(), d_output, imgSize, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_workspace);

    return output;
}

} // namespace Filters
