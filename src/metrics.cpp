#include "metrics.hpp"
#include <cmath>
#include <algorithm>

namespace Metrics {

    //peak signal to noise ratio. higher is better
float calculatePSNR(const Image& original, const Image& denoised) {
    //matching dims
    if (original.getWidth() != denoised.getWidth() ||
        original.getHeight() != denoised.getHeight() ||
        original.getChannels() != denoised.getChannels()) {
        return -1.0f;
    }

    double mse = 0.0;
    int width = original.getWidth();
    int height = original.getHeight();
    int channels = original.getChannels();
    int total_pixels = width * height * channels;

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float diff = original.getPixel(x, y, c) - denoised.getPixel(x, y, c);
                mse += diff * diff;
            }
        }
    }

    mse /= total_pixels; //sum of se to mse

    if (mse < 1e-10) return 99.0f; // Max PSNR for identical images

    //assumes normalized img so maxval is 1.0
    float max_val = 1.0f;
    return 10.0f * std::log10((max_val * max_val) / mse);
}

//structural similarity index measure
//this is global SSIM, have to change this later
float calculateSSIM(const Image& original, const Image& denoised) {
    //matching dims
    if (original.getWidth() != denoised.getWidth() ||
        original.getHeight() != denoised.getHeight() ||
        original.getChannels() != denoised.getChannels()) {
        return -1.0f;
    }
    
    //computing means
    double meanX = 0.0f;
    double meanY = 0.0f;
    int total_pixels = original.getWidth() * original.getHeight() * original.getChannels();

    for (int c = 0; c < original.getChannels(); c++) {
        for (int y = 0; y < original.getHeight(); y++) {
            for (int x = 0; x < original.getWidth(); x++) {
                meanX += original.getPixel(x, y, c);
                meanY += denoised.getPixel(x, y, c);
            }
        }
    }
    meanX /= total_pixels;
    meanY /= total_pixels;

    //computing variances and covariance
    double varX = 0.0f;
    double varY = 0.0f;
    double covXY = 0.0f;

    for (int c = 0; c < original.getChannels(); c++) {
        for (int y = 0; y < original.getHeight(); y++) {
            for (int x = 0; x < original.getWidth(); x++) {
                varX += (original.getPixel(x, y, c) - meanX) * (original.getPixel(x, y, c) - meanX);
                varY += (denoised.getPixel(x, y, c) - meanY) * (denoised.getPixel(x, y, c) - meanY);
                covXY += (original.getPixel(x, y, c) - meanX) * (denoised.getPixel(x, y, c) - meanY);
            }
        }
    }
    varX /= total_pixels;
    varY /= total_pixels;
    covXY /= total_pixels;

    //constants
    //should be c1 = (0.01*L)*(0.01*L) and c2 = (0.03*L)*(0.03*L), but since pixels here are normalized, L = 1
    double c1 = 0.0001;
    double c2 = 0.0009;

    //ssim computation
    double numerator = (2 * (meanX * meanY) + c1) * (2 * covXY + c2);
    double denominator = ((meanX * meanX) + (meanY * meanY) + c1) * (varX + varY + c2);

    return numerator / denominator;
}

}
