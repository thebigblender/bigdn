#include <iostream>
#include <string>
#include "benchmark.hpp"

int main(int argc, char* argv[]) {
    // Default image in project root
    std::string imagePath = "input.png";
    int kernelSize = 15;
    int numRuns = 5;

    if (argc > 1) {
        imagePath = argv[1];
    }
    if (argc > 2) {
        try {
            kernelSize = std::stoi(argv[2]);
            if (kernelSize % 2 == 0 || kernelSize <= 0) {
                std::cerr << "Warning: Kernel size should be a positive odd integer. Using default: 5" << std::endl;
                kernelSize = 5;
            }
        } catch (...) {
            std::cerr << "Warning: Invalid kernel size. Using default: 5" << std::endl;
            kernelSize = 5;
        }
    }
    if (argc > 3) {
        try {
            numRuns = std::stoi(argv[3]);
            if (numRuns <= 0) {
                std::cerr << "Warning: Number of runs should be positive. Using default: 10" << std::endl;
                numRuns = 10;
            }
        } catch (...) {
            std::cerr << "Warning: Invalid number of runs. Using default: 10" << std::endl;
            numRuns = 10;
        }
    }

    Benchmark::run(imagePath, kernelSize, numRuns);
    return 0;
}
