#include "metrics.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

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

    if (mse < 1e-10) return 99.0f;

    //assumes normalized img so maxval is 1.0
    float max_val = 1.0f;
    return 10.0f * std::log10((max_val * max_val) / mse);
}

//downsample image by 2x using box filter
static Image downsample(const Image& img) {
    int w = img.getWidth() / 2;
    int h = img.getHeight() / 2;
    int c = img.getChannels();
    if (w == 0) w = 1;
    if (h == 0) h = 1;
    Image out(w, h, c);

    for (int ch = 0; ch < c; ++ch) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float val = 0.0f;
                int count = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        int px = 2 * x + dx;
                        int py = 2 * y + dy;
                        if (px < img.getWidth() && py < img.getHeight()) {
                            val += img.getPixel(px, py, ch);
                            count++;
                        }
                    }
                }
                out.setPixel(x, y, ch, val / count);
            }
        }
    }
    return out;
}

//multi scale ssim calculation
float calculateMSSIM(const Image& original, const Image& denoised) {
    //matching dims
    if (original.getWidth() != denoised.getWidth() ||
        original.getHeight() != denoised.getHeight() ||
        original.getChannels() != denoised.getChannels()) {
        return -1.0f;
    }

    double msssim = 1.0;
    const double weights[5] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
    Image imgX = original;
    Image imgY = denoised;

    for (int j = 0; j < 5; ++j) {
        //means
        double meanX = 0.0;
        double meanY = 0.0;
        int total_pixels = imgX.getWidth() * imgX.getHeight() * imgX.getChannels();

        for (int ch = 0; ch < imgX.getChannels(); ch++) {
            for (int y = 0; y < imgX.getHeight(); y++) {
                for (int x = 0; x < imgX.getWidth(); x++) {
                    meanX += imgX.getPixel(x, y, ch);
                    meanY += imgY.getPixel(x, y, ch);
                }
            }
        }
        meanX /= total_pixels;
        meanY /= total_pixels;

        //variances and covariance
        double varX = 0.0;
        double varY = 0.0;
        double covXY = 0.0;

        for (int ch = 0; ch < imgX.getChannels(); ch++) {
            for (int y = 0; y < imgX.getHeight(); y++) {
                for (int x = 0; x < imgX.getWidth(); x++) {
                    double dx = imgX.getPixel(x, y, ch) - meanX;
                    double dy = imgY.getPixel(x, y, ch) - meanY;
                    varX += dx * dx;
                    varY += dy * dy;
                    covXY += dx * dy;
                }
            }
        }
        varX /= total_pixels;
        varY /= total_pixels;
        covXY /= total_pixels;

        double stdX = std::sqrt(std::max(0.0, varX));
        double stdY = std::sqrt(std::max(0.0, varY));

        double c1 = 0.0001;
        double c2 = 0.0009;
        double c3 = c2 / 2.0;

        //l, c, s terms
        double l = (2 * meanX * meanY + c1) / (meanX * meanX + meanY * meanY + c1);
        double c = (2 * stdX * stdY + c2) / (varX + varY + c2);
        double s = (covXY + c3) / (stdX * stdY + c3);

        if (l < 0) l = 0;
        if (c < 0) c = 0;
        if (s < 0) s = 0;

        //fractional powers and scale iteration
        if (j < 4 && imgX.getWidth() > 2 && imgX.getHeight() > 2) {
            msssim *= std::pow(c, weights[j]) * std::pow(s, weights[j]);
            imgX = downsample(imgX);
            imgY = downsample(imgY);
        } else {
            msssim *= std::pow(l, weights[j]) * std::pow(c, weights[j]) * std::pow(s, weights[j]);
            break;
        }
    }

    return static_cast<float>(msssim);
}

} // namespace Metrics
