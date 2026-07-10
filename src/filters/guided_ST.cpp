#include "filters.hpp"
#include <algorithm>
#include <vector>

namespace Filters {

//sliding window box filter
static Image boxFilter(const Image& input, int kernelSize) {
    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    const int planeSize = width * height;
    
    //horizontal pass
    Image temp(width, height, channels);
    const int radius = kernelSize / 2;
    const float invSize = 1.0f / kernelSize;

    const float* inData = input.getData();
    float* tempData = temp.getData();

    for (int c = 0; c < channels; ++c) {
        const int channelOffset = c * planeSize;
        for (int y = 0; y < height; ++y) {
            const int rowOffset = y * width;
            
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int clampedX = std::max(0, std::min(k, width - 1));
                sum += inData[channelOffset + rowOffset + clampedX];
            }
            tempData[channelOffset + rowOffset + 0] = sum * invSize;

            //sliding window
            for (int x = 1; x < width; ++x) {
                int enteringX = std::min(x + radius, width - 1);
                int leavingX = std::max(0, x - 1 - radius);
                sum += inData[channelOffset + rowOffset + enteringX] - inData[channelOffset + rowOffset + leavingX];
                tempData[channelOffset + rowOffset + x] = sum * invSize;
            }
        }
    }

    //vertical pass
    Image output(width, height, channels);
    float* outData = output.getData();

    for (int c = 0; c < channels; ++c) {
        const int channelOffset = c * planeSize;
        for (int x = 0; x < width; ++x) {
            
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int clampedY = std::max(0, std::min(k, height - 1));
                sum += tempData[channelOffset + clampedY * width + x];
            }
            outData[channelOffset + 0 + x] = sum * invSize;

            for (int y = 1; y < height; ++y) {
                int enteringY = std::min(y + radius, height - 1);
                int leavingY = std::max(0, y - 1 - radius);
                sum += tempData[channelOffset + enteringY * width + x] - tempData[channelOffset + leavingY * width + x];
                outData[channelOffset + y * width + x] = sum * invSize;
            }
        }
    }

    return output;
}

Image guidedCPU_ST(const Image& input, const Image& guidance, int kernelSize, float eps) {
    const int width = input.getWidth();
    const int height = input.getHeight();
    const int channels = input.getChannels();
    const int planeSize = width * height;
    const int totalPixels = planeSize * channels;

    //local means
    Image mean_I = boxFilter(guidance, kernelSize);
    Image mean_p = boxFilter(input, kernelSize);

    //local correlations
    Image I_sq(width, height, channels);
    Image I_p(width, height, channels);
    
    const float* inData = input.getData();
    const float* guideData = guidance.getData();
    float* I_sq_ptr = I_sq.getData();
    float* I_p_ptr = I_p.getData();

    //helper images
    for (int i = 0; i < totalPixels; ++i) {
        float i_val = guideData[i];
        float p_val = inData[i];
        I_sq_ptr[i] = i_val * i_val;
        I_p_ptr[i] = i_val * p_val;
    }

    Image mean_II = boxFilter(I_sq, kernelSize);
    Image mean_Ip = boxFilter(I_p, kernelSize);

    //computing var, covar, and coeffs a and b
    Image a(width, height, channels);
    Image b(width, height, channels);

    const float* m_I_ptr = mean_I.getData();
    const float* m_p_ptr = mean_p.getData();
    const float* m_II_ptr = mean_II.getData();
    const float* m_Ip_ptr = mean_Ip.getData();
    float* a_ptr = a.getData();
    float* b_ptr = b.getData();

    for (int i = 0; i < totalPixels; ++i) {
        float m_I = m_I_ptr[i];
        float m_p = m_p_ptr[i];
        float m_II = m_II_ptr[i];
        float m_Ip = m_Ip_ptr[i];

        float var_I = m_II - m_I * m_I;
        float cov_Ip = m_Ip - m_I * m_p;

        float a_val = cov_Ip / (var_I + eps);
        float b_val = m_p - a_val * m_I;

        a_ptr[i] = a_val;
        b_ptr[i] = b_val;
    }

    //avging the coeffs over local window
    Image mean_a = boxFilter(a, kernelSize);
    Image mean_b = boxFilter(b, kernelSize);

    //final
    Image output(width, height, channels);
    const float* m_a_ptr = mean_a.getData();
    const float* m_b_ptr = mean_b.getData();
    float* outData = output.getData();

    for (int i = 0; i < totalPixels; ++i) {
        outData[i] = m_a_ptr[i] * guideData[i] + m_b_ptr[i];
    }

    return output;
}

}
