#include "filters.hpp"
#include <algorithm>

namespace Filters {

Image meanCPU_OMP(const Image& input, int kernelSize)
{
    if (kernelSize <= 1)
        return input;

    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();

    Image output(width, height, channels);

    const int radius = kernelSize / 2;
    const float invArea = 1.0f / (kernelSize * kernelSize);

    const float* inData = input.getData();
    float* outData = output.getData();
    const int channelStride = width * height;

    //interior pixels only
    for (int c = 0; c < channels; c++)
    {
        #pragma omp parallel for schedule(static)
        for (int y = radius; y < height - radius; y++)
        {
            const int c_offset = c * channelStride;
            for (int x = radius; x < width - radius; x++)
            {
                float sum = 0.0f;

                for (int ky = -radius; ky <= radius; ky++)
                {
                    const int rowOffset = c_offset + (y + ky) * width;
                    for (int kx = -radius; kx <= radius; kx++)
                    {
                        sum += inData[rowOffset + (x + kx)];
                    }
                }

                outData[c_offset + y * width + x] = sum * invArea;
            }
        }
    }

    //border pixels
    #pragma omp parallel for schedule(static) collapse(2)
    for (int c = 0; c < channels; c++)
    {
        for (int y = 0; y < height; y++)
        {
            const int c_offset = c * channelStride;
            for (int x = 0; x < width; x++)
            {
                if (x >= radius &&
                    x < width - radius &&
                    y >= radius &&
                    y < height - radius)
                {
                    continue;
                }

                float sum = 0.0f;

                for (int ky = -radius; ky <= radius; ky++)
                {
                    int ny = std::max(0, std::min(y + ky, height - 1));
                    const int rowOffset = c_offset + ny * width;
                    for (int kx = -radius; kx <= radius; kx++)
                    {
                        int nx = std::max(0, std::min(x + kx, width - 1));
                        sum += inData[rowOffset + nx];
                    }
                }

                outData[c_offset + y * width + x] = sum * invArea;
            }
        }
    }

    return output;
}

}
