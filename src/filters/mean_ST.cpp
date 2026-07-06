#include "filters.hpp"
#include <algorithm>

namespace Filters {

Image meanCPU_ST(const Image& input, int kernelSize) {
    if (kernelSize <= 1) {
        return input;
    }

    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    Image output(width, height, channels);

    int halfSize = kernelSize / 2;

    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;

                for (int ky = -halfSize; ky <= halfSize; ++ky) {
                    for (int kx = -halfSize; kx <= halfSize; ++kx) {
                        sum += input.getPixelClamped(x + kx, y + ky, c);
                    }
                }

                float avg = sum / (kernelSize * kernelSize);
                output.setPixel(x, y, c, avg);
            }
        }
    }

    return output;
}

}
