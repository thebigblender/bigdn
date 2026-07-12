#pragma once
#include "image.hpp"

namespace Metrics {
    float calculatePSNR(const Image& original, const Image& denoised);
    float calculateMSSIM(const Image& original, const Image& denoised);
}
