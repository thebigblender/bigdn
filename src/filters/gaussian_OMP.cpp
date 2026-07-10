#include "filters.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace Filters {

Image gaussianCPU_OMP(const Image& input, int kernelSize, float sigma)
{
    if (kernelSize <= 1 || sigma <= 0.0f) {
        return input;
    }

    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    Image output(width, height, channels);

    const int radius = kernelSize / 2;

    //precompute and normalize the 2D Gaussian Kernel
    std::vector<float> kernel(kernelSize * kernelSize);
    float kernelSum = 0.0f;
    const float twoSigmaSq = 2.0f * sigma * sigma;

    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            const float distSq = static_cast<float>(kx * kx + ky * ky);
            const float weight = std::exp(-distSq / twoSigmaSq);
            const int kernelIdx = (ky + radius) * kernelSize + (kx + radius);
            kernel[kernelIdx] = weight;
            kernelSum += weight;
        }
    }

    const float invKernelSum = 1.0f / kernelSum;
    for (float& weight : kernel) {
        weight *= invKernelSum;
    }

    //getting raw ptrs
    const float* inData = input.getData();
    float* outData = output.getData();

    const int channelStride = width * height;

    //precompute row offsets
    const std::vector<int> rowOffsets = [&]() {
        std::vector<int> offsets(height);
        for (int y = 0; y < height; ++y) {
            offsets[y] = y * width;
        }
        return offsets;
    }();

    //le filter - interior
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < channels; ++c)
    {
        const int channelOffset = c * channelStride;

        for (int y = radius; y < height - radius; ++y)
        {
            const int outRowOffset = rowOffsets[y];

            for (int x = radius; x < width - radius; ++x)
            {
                float sum = 0.0f;

                for (int ky = -radius; ky <= radius; ++ky)
                {
                    const int rowOffset = rowOffsets[y + ky];
                    const int kernelRow = (ky + radius) * kernelSize;

                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        const float pixel = inData[channelOffset + rowOffset + (x + kx)];
                        sum += pixel * kernel[kernelRow + kx + radius];
                    }
                }

                outData[channelOffset + outRowOffset + x] = sum;
            }
        }
    }

    //le filter
    #pragma omp parallel for schedule(static) collapse(2)
    for (int c = 0; c < channels; ++c)
    {
        for (int y = 0; y < height; ++y)
        {
            const int channelOffset = c * channelStride;
            const int outRowOffset = rowOffsets[y];

            for (int x = 0; x < width; ++x)
            {
                //skip if already processed in interior step
                if (x >= radius && x < width - radius &&
                    y >= radius && y < height - radius)
                {
                    continue;
                }

                float sum = 0.0f;

                for (int ky = -radius; ky <= radius; ++ky)
                {
                    const int clampedY = std::max(0, std::min(y + ky, height - 1));
                    const int rowOffset = rowOffsets[clampedY];
                    const int kernelRow = (ky + radius) * kernelSize;

                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        const int clampedX = std::max(0, std::min(x + kx, width - 1));

                        const float pixel = inData[channelOffset + rowOffset + clampedX];
                        sum += pixel * kernel[kernelRow + kx + radius];
                    }
                }

                outData[channelOffset + outRowOffset + x] = sum;
            }
        }
    }

    return output;
}

}
