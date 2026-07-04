#pragma once
#include "image.hpp"

namespace Filters {
    Image meanCPU(const Image& input, int kernelSize);

    Image gaussianNaiveCPU(const Image& input, int kernelSize, float sigma);

    Image gaussianSeparableCPU(const Image& input, int kernelSize, float sigma);

    Image medianCPU(const Image& input, int kernelSize);

    Image bilateralCPU(const Image& input, int kernelSize, float sigmaSpatial, float sigmaRange);

    Image guidedCPU(const Image& input, const Image& guidance, int kernelSize, float eps);

    Image nlmCPU(const Image& input, int searchWindowSize, int patchSize, float h);
}
