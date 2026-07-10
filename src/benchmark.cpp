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

// Helper function to generate a synthetic image with noise for benchmarking
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

    // Load image
    if (!input.load(imagePath)) {
        std::string altPath = "../" + imagePath;
        if (!input.load(altPath)) {
            std::cerr << "Error: Failed to load image: " << imagePath << std::endl;
            exit(1);
        }
    }

    Image outputST(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGaussianST(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGuidedST(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGuidedOMP(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGuidedCUDA(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputOMP(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGaussianOMP(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputCUDA(input.getWidth(), input.getHeight(), input.getChannels());
    Image outputGaussianCUDA(input.getWidth(), input.getHeight(), input.getChannels());

    std::cout << "\nStarting nanobench benchmarks..." << std::endl;

    // Set up nanobench
    ankerl::nanobench::Bench bench;
    bench.title("Mean Filter Denoising")
         .unit("evaluation")
         .relative(true)
         .epochs(5)
         .minEpochTime(std::chrono::milliseconds(100))
         .output(nullptr); // Suppress default wide console table

    if (numRuns > 0) {
        bench.minEpochIterations(numRuns);
    }

    // Run Benchmarks
    bench.run("Mean ST", [&] {
        outputST = Filters::meanCPU_ST(input, kernelSize);
        ankerl::nanobench::doNotOptimizeAway(outputST.getData());
    });

    bench.run("Mean OMP", [&] {
        outputOMP = Filters::meanCPU_OMP(input, kernelSize);
        ankerl::nanobench::doNotOptimizeAway(outputOMP.getData());
    });

    // Allocate GPU memory and copy input once before the benchmark loop
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    size_t imgSize = width * height * channels * sizeof(float);

    int totalPixels = width * height * channels;
    float* d_workspace = nullptr;
    cudaMalloc(&d_workspace, 11 * imgSize);

    float* d_input = d_workspace;
    float* d_temp1 = d_workspace + totalPixels;
    float* d_temp2 = d_workspace + 2 * totalPixels;
    float* d_output_mean = d_workspace + 3 * totalPixels;
    float* d_output_gaussian = d_workspace + 4 * totalPixels;
    float* d_output_guided = d_workspace + 5 * totalPixels;
    float* d_mean_I = d_workspace + 6 * totalPixels;
    float* d_mean_II = d_workspace + 7 * totalPixels;
    float* d_mean_Ip = d_workspace + 8 * totalPixels;
    float* d_a = d_workspace + 9 * totalPixels;
    float* d_b = d_workspace + 10 * totalPixels;

    // Compute and copy 1D Gaussian kernel to GPU constant memory
    const int radius = kernelSize / 2;
    std::vector<float> h_kernel(kernelSize);
    float kernelSum = 0.0f;
    const float twoSigmaSq = 2.0f * 1.5f * 1.5f; // sigma = 1.5f
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

    bench.run("Guided ST", [&] {
        outputGuidedST = Filters::guidedCPU_ST(input, input, kernelSize, 0.04f);
        ankerl::nanobench::doNotOptimizeAway(outputGuidedST.getData());
    });

    bench.run("Guided OMP", [&] {
        outputGuidedOMP = Filters::guidedCPU_OMP(input, input, kernelSize, 0.04f);
        ankerl::nanobench::doNotOptimizeAway(outputGuidedOMP.getData());
    });

    bench.run("Guided CUDA", [&] {
        Filters::guidedCUDA_NoAlloc(d_input, d_input, d_output_guided, d_temp1, d_temp2,
                                    d_mean_I, d_mean_II, d_mean_Ip,
                                    d_a, d_b,
                                    width, height, channels, kernelSize, 0.04f);
        cudaDeviceSynchronize();
    });

    // Copy separable outputs back to host and clean up GPU memory
    cudaMemcpy(outputCUDA.getData(), d_output_mean, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputGaussianCUDA.getData(), d_output_gaussian, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputGuidedCUDA.getData(), d_output_guided, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_workspace);

    // Save Output Images
    outputST.save("output_ST.png");
    outputGaussianST.save("output_gaussian_ST.png");
    outputGuidedST.save("output_guided_ST.png");
    outputGuidedOMP.save("output_guided_OMP.png");
    outputGuidedCUDA.save("output_guided_CUDA.png");
    outputOMP.save("output_OMP.png");
    outputGaussianOMP.save("output_gaussian_OMP.png");
    outputCUDA.save("output_CUDA.png");
    outputGaussianCUDA.save("output_gaussian_CUDA.png");

    // Print Benchmark Summary Table (including Kernel Size and Cycles)
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

    // Print Quality Metrics (All relative to the original noisy input image)
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << " DENOISING QUALITY METRICS (Relative to original noisy input)\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left << std::setw(25) << " Implementation"
              << std::right << std::setw(18) << "PSNR (dB)"
              << std::setw(18) << "SSIM" << "\n";
    std::cout << std::string(70, '-') << "\n";

    auto printMetricRow = [&](const std::string& name, const Image& output) {
        float psnr = Metrics::calculatePSNR(input, output);
        float ssim = Metrics::calculateSSIM(input, output);
        std::cout << std::left << " " << std::setw(24) << name
                  << std::right << std::fixed << std::setprecision(2) << std::setw(18) << psnr
                  << std::setprecision(4) << std::setw(18) << ssim << "\n";
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
    std::cout << std::string(70, '=') << "\n\n";
}

} // namespace Benchmark
