#pragma once
#include "image.hpp"

namespace Metrics {
    float calculatePSNR(const Image& original, const Image& denoised);
    float calculateSSIM(const Image& original, const Image& denoised);
}
