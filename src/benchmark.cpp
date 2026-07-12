#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "benchmark.hpp"
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <random>

#include "image.hpp"
#include "filters.hpp"
#include "metrics.hpp"

namespace Benchmark {

// helper to generate synthetic image with noise
Image generateSyntheticImage(int width, int height) {
    Image img(width, height, 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float r = static_cast<float>(x) / width;
            float g = static_cast<float>(y) / height;
            float b = 0.5f * (r + g);

            if (((x / 32) + (y / 32)) % 2 == 0) {
                r = 1.0f - r;
                g = 1.0f - g;
                b = 1.0f - b;
            }

            img.setPixel(x, y, 0, r);
            img.setPixel(x, y, 1, g);
            img.setPixel(x, y, 2, b);
        }
    }

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.08f);

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float val = img.getPixel(x, y, c);
                val += dist(gen);
                val = std::max(0.0f, std::min(val, 1.0f));
                img.setPixel(x, y, c, val);
            }
        }
    }

    return img;
}

void run(const std::string& imagePath, int kernelSize, int numRuns) {
    Image input;

    // load image
    if (!input.load(imagePath)) {
        std::string altPath = "../" + imagePath;
        if (!input.load(altPath)) {
            std::cerr << "Error: Failed to load image: " << imagePath << std::endl;
            exit(1);
        }
    }

    // load gt image
    Image gt;
    bool has_gt = true;
    if (!gt.load("TEST_GT.png")) {
        std::string altPath = "../TEST_GT.png";
        if (!gt.load(altPath)) {
            std::cout << "Warning: Failed to load GT image: TEST_GT.png. Falling back to noisy input for metrics." << std::endl;
            gt = input;
            has_gt = false;
        }
    }

    Image outputST(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGaussianST(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGuidedST(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGuidedOMP(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGuidedCUDA(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputJointGuidedCUDA(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputATrousCUDA(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputOMP(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGaussianOMP(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputCUDA(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGaussianCUDA(input.getWidth(), input.getHeight(), input.getChannels());

    std::cout << "\nStarting nanobench benchmarks..." << std::endl;

    // set up nanobench
    ankerl::nanobench::Bench bench;
    bench.title("Mean Filter Denoising")
         .unit("evaluation")
         .relative(true)
         .epochs(5)
         .minEpochTime(std::chrono::milliseconds(100))
         .output(nullptr);

    if (numRuns > 0) {
        bench.minEpochIterations(numRuns);
    }

    // run benchmarks
    bench.run("Mean ST", [&] {
        outputST = Filters::meanCPU_ST(input, kernelSize);
        ankerl::nanobench::doNotOptimizeAway(outputST.getData());
    });

    bench.run("Mean OMP", [&] {
        outputOMP = Filters::meanCPU_OMP(input, kernelSize);
        ankerl::nanobench::doNotOptimizeAway(outputOMP.getData());
    });

    // allocate gpu memory
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    size_t imgSize = width * height * channels * sizeof(float);

    int totalPixels = width * height * channels;
    float* d_workspace = nullptr;
    cudaMalloc(&d_workspace, 15 * imgSize);

    float* d_input          = d_workspace;
    float* d_normal         = d_workspace + totalPixels;
    float* d_albedo         = d_workspace + 2 * totalPixels;
    float* d_output_mean    = d_workspace + 3 * totalPixels;
    float* d_output_gaussian= d_workspace + 4 * totalPixels;
    float* d_output_guided  = d_workspace + 5 * totalPixels;
    float* d_output_jg      = d_workspace + 6 * totalPixels;
    float* d_shading        = d_workspace + 7 * totalPixels;
    float* d_temp1          = d_workspace + 8 * totalPixels;
    float* d_temp2          = d_workspace + 9 * totalPixels;
    
    // guided filter temps
    float* d_mean_I         = d_workspace + 10 * totalPixels;
    float* d_mean_II        = d_workspace + 11 * totalPixels;
    float* d_mean_Ip        = d_workspace + 12 * totalPixels;
    float* d_a              = d_workspace + 13 * totalPixels;
    float* d_b              = d_workspace + 14 * totalPixels;

    // load normal and albedo images
    Image normal;
    if (!normal.load("TEST_NORMAL.png")) {
        std::string altPath = "../TEST_NORMAL.png";
        if (!normal.load(altPath)) {
            std::cerr << "Error: Failed to load normal image: TEST_NORMAL.png" << std::endl;
            exit(1);
        }
    }

    Image albedo;
    if (!albedo.load("TEST_ALBEDO.png")) {
        std::string altPath = "../TEST_ALBEDO.png";
        if (!albedo.load(altPath)) {
            std::cerr << "Error: Failed to load albedo image: TEST_ALBEDO.png" << std::endl;
            exit(1);
        }
    }

    if (normal.getWidth() != width || normal.getHeight() != height || normal.getChannels() != channels) {
        std::cerr << "Error: Normal image dimensions (" << normal.getWidth() << "x" << normal.getHeight() 
                  << ") mismatch input (" << width << "x" << height << ")" << std::endl;
        exit(1);
    }
    if (albedo.getWidth() != width || albedo.getHeight() != height || albedo.getChannels() != channels) {
        std::cerr << "Error: Albedo image dimensions (" << albedo.getWidth() << "x" << albedo.getHeight() 
                  << ") mismatch input (" << width << "x" << height << ")" << std::endl;
        exit(1);
    }

    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, normal.getData(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo, albedo.getData(), imgSize, cudaMemcpyHostToDevice);

    // compute and copy 1d gaussian kernel
    const int radius = kernelSize / 2;
    std::vector<float> h_kernel(kernelSize);
    float kernelSum = 0.0f;
    const float twoSigmaSq = 2.0f * 1.5f * 1.5f;
    for (int i = 0; i < kernelSize; ++i) {
        float x = static_cast<float>(i - radius);
        h_kernel[i] = std::exp(-(x * x) / twoSigmaSq);
        kernelSum += h_kernel[i];
    }
    const float invSum = 1.0f / kernelSum;
    for (float& val : h_kernel) {
        val *= invSum;
    }

    cudaMemcpy(d_input, input.getData(), imgSize, cudaMemcpyHostToDevice);
    Filters::gaussianCUDA_UploadKernel(h_kernel.data(), kernelSize);

    bench.run("Mean CUDA", [&] {
        Filters::meanCUDA_NoAlloc(d_input, d_temp1, d_output_mean, width, height, channels, kernelSize);
        cudaDeviceSynchronize();
    });

    bench.run("Gaussian ST", [&] {
        outputGaussianST = Filters::gaussianCPU_ST(input, kernelSize, 1.5f);
        ankerl::nanobench::doNotOptimizeAway(outputGaussianST.getData());
    });

    bench.run("Gaussian OMP", [&] {
        outputGaussianOMP = Filters::gaussianCPU_OMP(input, kernelSize, 1.5f);
        ankerl::nanobench::doNotOptimizeAway(outputGaussianOMP.getData());
    });

    bench.run("Gaussian CUDA", [&] {
        Filters::gaussianCUDA_NoAlloc(d_input, d_temp1, d_output_gaussian, width, height, channels, kernelSize);
        cudaDeviceSynchronize();
    });

    // guided filter runs
    bench.run("Guided ST", [&] {
        outputGuidedST = Filters::guidedCPU_ST(input, normal, kernelSize, 0.005f);
        ankerl::nanobench::doNotOptimizeAway(outputGuidedST.getData());
    });

    bench.run("Guided OMP", [&] {
        outputGuidedOMP = Filters::guidedCPU_OMP(input, normal, kernelSize, 0.005f);
        ankerl::nanobench::doNotOptimizeAway(outputGuidedOMP.getData());
    });

    bench.run("Guided CUDA", [&] {
        Filters::guidedCUDA_NoAlloc(d_input, d_normal, d_output_guided, d_temp1, d_temp2,
                                    d_mean_I, d_mean_II, d_mean_Ip,
                                    d_a, d_b,
                                    width, height, channels, kernelSize, 0.005f);
        cudaDeviceSynchronize();
    });

    bench.run("Joint Guided CUDA", [&] {
        Filters::jointGuidedCUDA_NoAlloc(d_input, d_normal, d_albedo, d_output_jg,
                                        d_shading, d_temp1, d_temp2,
                                        d_mean_I, d_mean_II, d_mean_Ip,
                                        d_a, d_b,
                                        width, height, channels, kernelSize, 0.005f);
        cudaDeviceSynchronize();
    });

    // 5 passes of a-trous filter
    bench.run("A-Trous CUDA", [&] {
        Filters::aTrousWaveletCUDA_NoAlloc(d_input, d_normal, d_albedo, d_shading,
                                           d_temp1, d_temp2,
                                           width, height, channels, 5, 0.15f, 0.1f, 0.05f);
        cudaDeviceSynchronize();
    });

    // copy outputs back to host and clean up gpu memory
    cudaMemcpy(outputCUDA.getData(), d_output_mean, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputGaussianCUDA.getData(), d_output_gaussian, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputGuidedCUDA.getData(), d_output_guided, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputJointGuidedCUDA.getData(), d_output_jg, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputATrousCUDA.getData(), d_shading, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_workspace);

    // save output images
    outputST.save("output_ST.png");
    outputGaussianST.save("output_gaussian_ST.png");
    outputGuidedST.save("output_guided_ST.png");
    outputGuidedOMP.save("output_guided_OMP.png");
    outputGuidedCUDA.save("output_guided_CUDA.png");
    outputJointGuidedCUDA.save("output_joint_guided_CUDA.png");
    outputATrousCUDA.save("output_atrous_CUDA.png");
    outputOMP.save("output_OMP.png");
    outputGaussianOMP.save("output_gaussian_OMP.png");
    outputCUDA.save("output_CUDA.png");
    outputGaussianCUDA.save("output_gaussian_CUDA.png");

    // print benchmark summary table
    const auto& results = bench.results();
    if (!results.empty()) {
        auto formatTime = [](double sec) {
            double ms = sec * 1000.0;
            char buf[32];
            if (ms >= 1.0) {
                std::snprintf(buf, sizeof(buf), "%.3f ms", ms);
            } else {
                std::snprintf(buf, sizeof(buf), "%.3f us", ms * 1000.0);
            }
            return std::string(buf);
        };

        std::cout << "\n" << std::string(75, '=') << "\n";
        std::cout << " BIGDN benchmark (Kernel size: " << kernelSize << "\n";
        std::cout << " Img Dims: " << input.getWidth() << "x" << input.getHeight();
        std::cout << " Config:    " << results.front().config().mNumEpochs << " epochs, "
                  << results.front().config().mMinEpochIterations << " min it/epoch\n";
        std::cout << std::string(75, '-') << "\n";
        std::cout << std::left << std::setw(28) << " Thing"
                  << std::right << std::setw(16) << "Median Time"
                  << std::setw(13) << "Error %"
                  << std::setw(15) << "Speedup" << "\n";
        std::cout << std::string(75, '-') << "\n";

        double meanBaseline = results[0].median(ankerl::nanobench::Result::Measure::elapsed);
        double gaussianBaseline = (results.size() > 3) ? results[3].median(ankerl::nanobench::Result::Measure::elapsed) : meanBaseline;
        double guidedBaseline = (results.size() > 6) ? results[6].median(ankerl::nanobench::Result::Measure::elapsed) : meanBaseline;

        int idx = 0;
        for (const auto& r : results) {
            if (idx == 3 || idx == 6) {
                std::cout << std::string(75, '-') << "\n";
            }
            double medianSec = r.median(ankerl::nanobench::Result::Measure::elapsed);
            double errPercent = r.medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0;
            double baselineTime;
            if (idx < 3) {
                baselineTime = meanBaseline;
            } else if (idx < 6) {
                baselineTime = gaussianBaseline;
            } else {
                baselineTime = guidedBaseline;
            }
            double speedup = (medianSec > 0.0) ? (baselineTime / medianSec) : 0.0;

            std::cout << std::left << " " << std::setw(27) << r.config().mBenchmarkName
                      << std::right << std::setw(16) << formatTime(medianSec)
                      << std::fixed << std::setprecision(1) << std::setw(12) << errPercent << "%"
                      << std::setprecision(2) << std::setw(14) << speedup << "x" << "\n";
            idx++;
        }
        std::cout << std::string(75, '=') << "\n";
    }

    // print quality metrics
    std::cout << "\n" << std::string(70, '=') << "\n";
    if (has_gt) {
        std::cout << " DENOISING QUALITY METRICS (Relative to clean TEST_GT.png)\n";
    } else {
        std::cout << " DENOISING QUALITY METRICS (Relative to original noisy input)\n";
    }
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(25) << " Implementation"
              << std::right << std::setw(18) << "PSNR (dB)"
              << std::setw(18) << "MS-SSIM" << "\n";
    std::cout << std::string(70, '-') << "\n";

    auto printMetricRow = [&](const std::string& name, const Image& output) {
        float psnr = Metrics::calculatePSNR(gt, output);
        float msssim = Metrics::calculateMSSIM(gt, output);
        std::cout << std::left << " " << std::setw(24) << name
                  << std::right << std::fixed << std::setprecision(2) << std::setw(18) << psnr
                  << std::setprecision(4) << std::setw(18) << msssim << "\n";
    };

    printMetricRow("Single-Threaded (ST)", outputST);
    printMetricRow("OpenMP (OMP)", outputOMP);
    printMetricRow("CUDA (GPU)", outputCUDA);
    std::cout << std::string(70, '-') << "\n";
    printMetricRow("Gaussian ST", outputGaussianST);
    printMetricRow("Gaussian OMP", outputGaussianOMP);
    printMetricRow("Gaussian CUDA", outputGaussianCUDA);
    std::cout << std::string(70, '-') << "\n";
    printMetricRow("Guided ST", outputGuidedST);
    printMetricRow("Guided OMP", outputGuidedOMP);
    printMetricRow("Guided CUDA", outputGuidedCUDA);
    printMetricRow("Joint Guided CUDA", outputJointGuidedCUDA);
    printMetricRow("A-Trous CUDA", outputATrousCUDA);
    std::cout << std::string(70, '=') << "\n\n";
}

} // namespace Benchmark
