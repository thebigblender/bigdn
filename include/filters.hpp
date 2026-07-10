#pragma once
#include "image.hpp"

namespace Filters {
    Image meanCPU_ST(const Image& input, int kernelSize);
    Image meanCPU_OMP(const Image& input, int kernelSize);
    Image meanCUDA(const Image& input, int kernelSize);
    void meanCUDA_NoAlloc(const float* d_input, float* d_temp, float* d_output, int width, int height, int channels,
      int kernelSize);
    
    Image gaussianCPU_ST(const Image& input, int kernelSize, float sigma);
    Image gaussianCPU_OMP(const Image& input, int kernelSize, float sigma);
    Image gaussianSeparableCPU(const Image& input, int kernelSize, float sigma);
    Image gaussianCUDA(const Image& input, int kernelSize, float sigma);
    void gaussianCUDA_NoAlloc(const float* d_input, float* d_temp, float* d_output, int width, int height, int channels,
      int kernelSize);
    void gaussianCUDA_UploadKernel(const float* h_kernel, int kernelSize);
    
    Image medianCPU(const Image& input, int kernelSize);
    
    Image bilateralCPU(const Image& input, int kernelSize, float sigmaSpatial, float sigmaRange);

    Image guidedCPU(const Image& input, const Image& guidance, int kernelSize, float eps);

    Image nlmCPU(const Image& input, int searchWindowSize, int patchSize, float h);
}
