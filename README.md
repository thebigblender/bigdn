# bigdn

A benchmarking toolkit comparing various CPU, OpenMP, and CUDA implementations of classical image denoising algorithms, specifically targetting Monte-Carlo render noise.

# Known Issues

* Classical edge preserving filters (Guided, Joint Guided, A Trous) have a hard mathematical limit distinguishing high frequency noise from true details, which can leave splotchiness in high noise renders.
* The Joint Guided and A Trous filters are only supported on the CUDA device(blame my laziness).
* A Trous does not denoise salt and pepper noise well because pixel outlier spikes are treated as edges and are not smoothed(help pls).
* Joint Guided filter is prone to smudginess and detail loss.
* Gaussian filter is too smudgy and completely blurs geometric edges.

## Algorithms Implemented

* Mean Filter (CPU Single Threaded, OpenMP, CUDA)
* Gaussian Filter (CPU Single Threaded, OpenMP, CUDA)
* Guided Filter (CPU Single Threaded, OpenMP, CUDA)
* Joint Guided Filter (CUDA)
* A Trous Wavelet Filter (CUDA)

## Build Instructions

To build the project, run the CMake configuration and build commands from the root directory:

```bash
cmake -B build -S . -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
cmake --build build
```

This will configure CMake and compile the executable into the build directory.

## Running the Toolkit

Run the benchmark executable using the compiled binary.

### Options

* -h, --help            Show help message and exit
* -f, --filter <name>   Filter to run: mean, gaussian, guided, joint_guided, atrous, or benchmark (default: benchmark)
* -d, --device <name>   Execution device/implementation: cpu, omp, or cuda (default: cuda)
* -i, --input <path>    Noisy beauty input image path (default: TEST.png)
* -n, --normal <path>   Normal map input image path (default: TEST_NORMAL.png)
* -a, --albedo <path>   Albedo map input image path (default: TEST_ALBEDO.png)
* -o, --output <path>   Denoised output image path (default: denoised.png)
* -g, --gt <path>       Clean ground truth image path for metric computation (default: TEST_GT.png)
* -k, --kernel-size <N> Kernel size (must be a positive odd integer, default: 15)
* -p, --passes <N>      Number of A Trous passes (default: 5)
* -sc, --sigma-color <V> Color sigma parameter (default: 0.15)
* -sn, --sigma-normal <V> Normal sigma parameter (default: 0.1)
* -sa, --sigma-albedo <V> Albedo sigma parameter (default: 0.05)
* -r, --runs <N>        Number of iterations to run during benchmark mode (default: 5)

### Running Benchmark Mode

To run all filters sequentially and report timing and quality metric comparisons:

```bash
./build/bigdn -f benchmark
```

### Running a Specific Filter

To run the CUDA A Trous denoiser on custom images:

```bash
./build/bigdn -f atrous -i input.png -o output.png -p 5 -sc 0.15 -sn 0.1
```

## Denoising Metrics

Below are the quality metrics obtained relative to the clean reference image:

* Mean Filter: 28.23 dB PSNR, 0.9837 SSIM
* Gaussian Filter: 34.49 dB PSNR, 0.9963 SSIM
* Guided Filter: 28.20 dB PSNR, 0.9836 SSIM
* Joint Guided Filter 7x7 kernel: 32.7957 dB PSNR, 0.9944 SSIM 
* A Trous Wavelet Filter (4, 0.1, 0.15): 33.5894 dB PSNR, 0.9954 SSIM
