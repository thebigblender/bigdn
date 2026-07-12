#include <iostream>
#include <string>
#include <algorithm>
#include "image.hpp"
#include "filters.hpp"
#include "metrics.hpp"
#include "benchmark.hpp"

void printHelp() {
    std::cout << "Usage: ./build/bigdn [options]\n\n"
              << "Options:\n"
              << "  -h, --help            Show this help message and exit\n"
              << "  -f, --filter <name>   Filter to run: mean, gaussian, guided, joint_guided, atrous, or benchmark (default: benchmark)\n"
              << "  -d, --device <name>   Execution device/implementation: cpu, omp, or cuda (default: cuda)\n"
              << "  -i, --input <path>    Noisy beauty input image path (default: TEST.png)\n"
              << "  -n, --normal <path>   Normal map input image path (default: TEST_NORMAL.png)\n"
              << "  -a, --albedo <path>   Albedo map input image path (default: TEST_ALBEDO.png)\n"
              << "  -o, --output <path>   Denoised output image path (default: denoised.png)\n"
              << "  -g, --gt <path>       Clean ground truth image path for metric computation (default: TEST_GT.png)\n"
              << "  -k, --kernel-size <N> Kernel size (must be a positive odd integer, default: 15)\n"
              << "  -p, --passes <N>      Number of A Trous passes (default: 5)\n"
              << "  -sc, --sigma-color <V> Color sigma parameter (default: 0.15)\n"
              << "  -sn, --sigma-normal <V> Normal sigma parameter (default: 0.1)\n"
              << "  -sa, --sigma-albedo <V> Albedo sigma parameter (default: 0.05)\n"
              << "  -r, --runs <N>        Number of iterations to run during benchmark mode (default: 5)\n";
}

int main(int argc, char* argv[]) {
    std::string filter = "benchmark";
    std::string device = "cuda";
    std::string inputPath = "TEST.png";
    std::string normalPath = "TEST_NORMAL.png";
    std::string albedoPath = "TEST_ALBEDO.png";
    std::string outputPath = "denoised.png";
    std::string gtPath = "TEST_GT.png";
    int kernelSize = 15;
    int passes = 5;
    float sigmaColor = 0.15f;
    float sigmaNormal = 0.1f;
    float sigmaAlbedo = 0.05f;
    int numRuns = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp();
            return 0;
        } else if ((arg == "-f" || arg == "--filter") && i + 1 < argc) {
            filter = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device = argv[++i];
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            inputPath = argv[++i];
        } else if ((arg == "-n" || arg == "--normal") && i + 1 < argc) {
            normalPath = argv[++i];
        } else if ((arg == "-a" || arg == "--albedo") && i + 1 < argc) {
            albedoPath = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if ((arg == "-g" || arg == "--gt") && i + 1 < argc) {
            gtPath = argv[++i];
        } else if ((arg == "-k" || arg == "--kernel-size") && i + 1 < argc) {
            try {
                kernelSize = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid kernel size.\n";
                return 1;
            }
        } else if ((arg == "-p" || arg == "--passes") && i + 1 < argc) {
            try {
                passes = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid number of passes.\n";
                return 1;
            }
        } else if ((arg == "-sc" || arg == "--sigma-color") && i + 1 < argc) {
            try {
                sigmaColor = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid color sigma.\n";
                return 1;
            }
        } else if ((arg == "-sn" || arg == "--sigma-normal") && i + 1 < argc) {
            try {
                sigmaNormal = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid normal sigma.\n";
                return 1;
            }
        } else if ((arg == "-sa" || arg == "--sigma-albedo") && i + 1 < argc) {
            try {
                sigmaAlbedo = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid albedo sigma.\n";
                return 1;
            }
        } else if ((arg == "-r" || arg == "--runs") && i + 1 < argc) {
            try {
                numRuns = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Error: Invalid number of runs.\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            printHelp();
            return 1;
        }
    }

    if (kernelSize % 2 == 0 || kernelSize <= 0) {
        std::cerr << "Error: Kernel size must be a positive odd integer.\n";
        return 1;
    }
    if (passes <= 0) {
        std::cerr << "Error: Number of passes must be positive.\n";
        return 1;
    }
    if (numRuns <= 0) {
        std::cerr << "Error: Number of runs must be positive.\n";
        return 1;
    }

    // Normalize parameters
    std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);
    std::transform(device.begin(), device.end(), device.begin(), ::tolower);

    if (filter != "benchmark") {
        Image input;
        if (!input.load(inputPath)) {
            std::cerr << "Error: Failed to load input image: " << inputPath << "\n";
            return 1;
        }

        Image output;
        Image normal;
        Image albedo;

        // Load normal and albedo guide maps if required by the filter
        if (filter == "guided" || filter == "joint_guided" || filter == "atrous") {
            if (!normal.load(normalPath)) {
                std::cerr << "Error: Failed to load normal image: " << normalPath << "\n";
                return 1;
            }
        }
        if (filter == "joint_guided" || filter == "atrous") {
            if (!albedo.load(albedoPath)) {
                std::cerr << "Error: Failed to load albedo image: " << albedoPath << "\n";
                return 1;
            }
        }

        std::cout << "Running filter '" << filter << "' on device '" << device << "'...\n";

        if (filter == "mean") {
            if (device == "cpu") {
                output = Filters::meanCPU_ST(input, kernelSize);
            } else if (device == "omp") {
                output = Filters::meanCPU_OMP(input, kernelSize);
            } else if (device == "cuda") {
                output = Filters::meanCUDA(input, kernelSize);
            } else {
                std::cerr << "Error: Unsupported device '" << device << "' for mean filter.\n";
                return 1;
            }
        } else if (filter == "gaussian") {
            if (device == "cpu") {
                output = Filters::gaussianCPU_ST(input, kernelSize, 1.5f);
            } else if (device == "omp") {
                output = Filters::gaussianCPU_OMP(input, kernelSize, 1.5f);
            } else if (device == "cuda") {
                output = Filters::gaussianCUDA(input, kernelSize, 1.5f);
            } else {
                std::cerr << "Error: Unsupported device '" << device << "' for gaussian filter.\n";
                return 1;
            }
        } else if (filter == "guided") {
            if (device == "cpu") {
                output = Filters::guidedCPU_ST(input, normal, kernelSize, 0.005f);
            } else if (device == "omp") {
                output = Filters::guidedCPU_OMP(input, normal, kernelSize, 0.005f);
            } else if (device == "cuda") {
                output = Filters::guidedCUDA(input, normal, kernelSize, 0.005f);
            } else {
                std::cerr << "Error: Unsupported device '" << device << "' for guided filter.\n";
                return 1;
            }
        } else if (filter == "joint_guided") {
            if (device != "cuda") {
                std::cout << "Warning: Joint Guided filter is CUDA-only. Falling back to CUDA.\n";
                device = "cuda";
            }
            output = Filters::jointGuidedCUDA(input, normal, albedo, kernelSize, 0.005f);
        } else if (filter == "atrous") {
            if (device != "cuda") {
                std::cout << "Warning: A-Trous Wavelet filter is CUDA-only. Falling back to CUDA.\n";
                device = "cuda";
            }
            output = Filters::aTrousWaveletCUDA(input, normal, albedo, passes, sigmaColor, sigmaNormal, sigmaAlbedo);
        } else {
            std::cerr << "Error: Unknown filter type '" << filter << "'.\n";
            printHelp();
            return 1;
        }

        if (output.save(outputPath)) {
            std::cout << "Denoised image successfully saved to: " << outputPath << "\n";
        } else {
            std::cerr << "Error: Failed to save output image: " << outputPath << "\n";
            return 1;
        }

        // Print metrics if ground truth is available
        Image gt;
        if (gt.load(gtPath)) {
            float psnr = Metrics::calculatePSNR(gt, output);
            float ssim = Metrics::calculateSSIM(gt, output);
            std::cout << "\nQuality Metrics (relative to " << gtPath << "):\n"
                      << "  PSNR: " << psnr << " dB\n"
                      << "  SSIM: " << ssim << "\n";
        }
    } else {
        std::cout << "Running benchmark mode on: " << inputPath << "\n";
        Benchmark::run(inputPath, kernelSize, numRuns);
    }
    return 0;
}
