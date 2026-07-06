#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "benchmark.hpp"

#include <iostream>
#include <chrono>
#include <vector>
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

    // Set up a simple grid pattern with gradients
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float r = static_cast<float>(x) / width;
            float g = static_cast<float>(y) / height;
            float b = 0.5f * (r + g);

            // Add checkerboard pattern
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

    // Add some synthetic Gaussian-like noise
    std::mt19937 gen(42); // Seeded generator for reproducibility
    std::normal_distribution<float> dist(0.0f, 0.08f); // mean = 0, stddev = 0.08

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float val = img.getPixel(x, y, c);
                val += dist(gen);
                val = std::max(0.0f, std::min(val, 1.0f)); // clamp to [0, 1]
                img.setPixel(x, y, c, val);
            }
        }
    }

    return img;
}

void run(const std::string& imagePath, int kernelSize, int numRuns) {
    Image input;
    bool loadedFromFile = false;

    // We check if the imagePath is loadable as provided.
    // If running from build/, the root directory is parent directory '../'.
    // So we try imagePath first, then check if '../' + imagePath is available.
    std::string pathAttempt = imagePath;
    std::cout << "Attempting to load image from: " << pathAttempt << std::endl;
    if (input.load(pathAttempt)) {
        loadedFromFile = true;
    } else {
        pathAttempt = "../" + imagePath;
        std::cout << "Failed. Attempting to load image from: " << pathAttempt << std::endl;
        if (input.load(pathAttempt)) {
            loadedFromFile = true;
        }
    }

    if (loadedFromFile) {
        std::cout << "Successfully loaded image: " << input.getWidth() << "x" << input.getHeight() 
                  << " with " << input.getChannels() << " channels." << std::endl;
    } else {
        std::cerr << "Failed to load image. Falling back to synthetic image." << std::endl;
        std::cout << "Generating 1024x1024 synthetic noisy image..." << std::endl;
        input = generateSyntheticImage(1024, 1024);
        std::cout << "Synthetic image generated." << std::endl;
    }

    Image outputST;
    Image outputOMP;
#ifdef WITH_CUDA
    Image outputCUDA;
#endif

    std::cout << "\nStarting nanobench benchmarks..." << std::endl;
    
    // Set up nanobench
    ankerl::nanobench::Bench bench;
    bench.title("Mean Filter Denoising (" + std::to_string(kernelSize) + "x" + std::to_string(kernelSize) + ")")
         .unit("evaluation")
         .relative(true) // Compares implementations relative to each other
         .epochs(2) // Keep it small (2 epochs) for fast benchmarking on large images
         .minEpochTime(std::chrono::milliseconds(1)); // Avoid running multiple iterations if it is slow

    if (numRuns > 0) {
        bench.minEpochIterations(numRuns);
    }

    // Benchmark Single-Threaded CPU implementation
    bench.run("Single-Threaded (ST)", [&] {
        outputST = Filters::meanCPU_ST(input, kernelSize);
        ankerl::nanobench::doNotOptimizeAway(outputST.getData());
    });

    // Benchmark OpenMP parallel implementation
    bench.run("OpenMP (OMP)", [&] {
        outputOMP = Filters::meanCPU_OMP(input, kernelSize);
        ankerl::nanobench::doNotOptimizeAway(outputOMP.getData());
    });

#ifdef WITH_CUDA
    // Benchmark CUDA GPU implementation
    bench.run("CUDA (GPU)", [&] {
        outputCUDA = Filters::meanCUDA(input, kernelSize);
        ankerl::nanobench::doNotOptimizeAway(outputCUDA.getData());
    });
#endif

    // Compute Metrics & Quality relative to input image
    float psnrST = Metrics::calculatePSNR(input, outputST);
    float ssimST = Metrics::calculateSSIM(input, outputST);
    
    float psnrOMP = Metrics::calculatePSNR(input, outputOMP);
    float ssimOMP = Metrics::calculateSSIM(input, outputOMP);

#ifdef WITH_CUDA
    float psnrCUDA = Metrics::calculatePSNR(input, outputCUDA);
    float ssimCUDA = Metrics::calculateSSIM(input, outputCUDA);
#endif

    // Compute Correctness differences (relative to Single-Threaded baseline)
    float psnrDiff = Metrics::calculatePSNR(outputST, outputOMP);
    float ssimDiff = Metrics::calculateSSIM(outputST, outputOMP);

#ifdef WITH_CUDA
    float psnrDiffCUDA = Metrics::calculatePSNR(outputST, outputCUDA);
    float ssimDiffCUDA = Metrics::calculateSSIM(outputST, outputCUDA);
#endif

    // Check if OpenMP output is identical to Single-Threaded (with minor tolerance)
    bool exactMatch = true;
    if (outputST.getWidth() == outputOMP.getWidth() &&
        outputST.getHeight() == outputOMP.getHeight() &&
        outputST.getChannels() == outputOMP.getChannels()) {
        const float* stData = outputST.getData();
        const float* ompData = outputOMP.getData();
        size_t totalElements = outputST.getWidth() * outputST.getHeight() * outputST.getChannels();
        for (size_t i = 0; i < totalElements; ++i) {
            if (std::abs(stData[i] - ompData[i]) > 1e-5f) {
                exactMatch = false;
                break;
            }
        }
    } else {
        exactMatch = false;
    }

#ifdef WITH_CUDA
    // Check if CUDA output is identical to Single-Threaded (with minor tolerance)
    bool exactMatchCUDA = true;
    if (outputST.getWidth() == outputCUDA.getWidth() &&
        outputST.getHeight() == outputCUDA.getHeight() &&
        outputST.getChannels() == outputCUDA.getChannels()) {
        const float* stData = outputST.getData();
        const float* cudaData = outputCUDA.getData();
        size_t totalElements = outputST.getWidth() * outputST.getHeight() * outputST.getChannels();
        for (size_t i = 0; i < totalElements; ++i) {
            if (std::abs(stData[i] - cudaData[i]) > 1e-5f) {
                exactMatchCUDA = false;
                break;
            }
        }
    } else {
        exactMatchCUDA = false;
    }
#endif

    // Save outputs relative to working directory
    std::string outSTPath = "output_ST.png";
    std::string outOMPPath = "output_OMP.png";
    outputST.save(outSTPath);
    outputOMP.save(outOMPPath);

#ifdef WITH_CUDA
    std::string outCUDAPath = "output_CUDA.png";
    outputCUDA.save(outCUDAPath);
#endif

    // Print Quality and Correctness Metrics
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << " QUALITY & CORRECTNESS METRICS (Kernel Size: " << kernelSize << "x" << kernelSize << ")\n";
    std::cout << " Image Dimensions: " << input.getWidth() << "x" << input.getHeight() 
              << " (" << input.getChannels() << " channels)\n";
    std::cout << std::string(60, '-') << "\n";
    
    std::cout << " Denoising Quality (PSNR/SSIM relative to noisy input):\n";
    std::cout << "   - ST Output:  PSNR = " << std::fixed << std::setprecision(2) << psnrST << " dB, SSIM = " << std::setprecision(4) << ssimST << "\n";
    std::cout << "   - OMP Output: PSNR = " << std::fixed << std::setprecision(2) << psnrOMP << " dB, SSIM = " << std::setprecision(4) << ssimOMP << "\n";
#ifdef WITH_CUDA
    std::cout << "   - CUDA Output: PSNR = " << std::fixed << std::setprecision(2) << psnrCUDA << " dB, SSIM = " << std::setprecision(4) << ssimCUDA << "\n";
#endif
    std::cout << std::string(60, '-') << "\n";

    std::cout << " Correctness Check (ST vs OMP):\n";
    std::cout << "   - PSNR: " << std::fixed << std::setprecision(2) << psnrDiff << " dB\n";
    std::cout << "   - SSIM: " << std::fixed << std::setprecision(4) << ssimDiff << "\n";
    std::cout << "   - Exact match (threshold 1e-5): " << (exactMatch ? "PASS" : "FAIL") << "\n";
#ifdef WITH_CUDA
    std::cout << std::string(60, '-') << "\n";
    std::cout << " Correctness Check (ST vs CUDA):\n";
    std::cout << "   - PSNR: " << std::fixed << std::setprecision(2) << psnrDiffCUDA << " dB\n";
    std::cout << "   - SSIM: " << std::fixed << std::setprecision(4) << ssimDiffCUDA << "\n";
    std::cout << "   - Exact match (threshold 1e-5): " << (exactMatchCUDA ? "PASS" : "FAIL") << "\n";
#endif
    std::cout << std::string(60, '-') << "\n";
    std::cout << " Saved outputs to:\n";
    std::cout << "   - Single-Threaded: " << outSTPath << "\n";
    std::cout << "   - OpenMP:          " << outOMPPath << "\n";
#ifdef WITH_CUDA
    std::cout << "   - CUDA:            " << outCUDAPath << "\n";
#endif
    std::cout << std::string(60, '=') << "\n\n";
}

} // namespace Benchmark
