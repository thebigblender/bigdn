# bigdn

A benchmarking toolkit comparing various CPU, OpenMP, and CUDA implementations of classical image denoising algorithms, specifically targetting Monte-Carlo render noise.

## Build Instructions

To build the project, run the CMake configuration and build commands from the root directory:

```bash
cmake -B build -S . -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
cmake --build build
```

This will configure CMake and compile the executable into the build directory.

## Running the Toolkit

Run the GUI or command line utility using the compiled binary.

### Running in GUI Mode

To launch the interactive split screen interface, run the executable with no arguments (or explicitly pass `--gui`):

```bash
./build/bigdn
```

### Running in CLI Mode

To run in CLI mode, specify a filter using the `-f` or `--filter` flag:

```bash
./build/bigdn -f <filter_name> [options]
```

### Options
```bash
-h, --help              #Show help message and exit
-f, --filter <name>     #Filter to run: mean, gaussian, guided, joint_guided, atrous, masked_atrous, or benchmark (default: benchmark)
-d, --device <name>     #Execution device/implementation: cpu, omp, or cuda (default: cuda)
-i, --input <path>      #Noisy beauty input image path (default: TEST.png)
-n, --normal <path>     #Normal map input image path (default: TEST_NORMAL.png)
-a, --albedo <path>     #Albedo map input image path (default: TEST_ALBEDO.png)
-o, --output <path>     #Denoised output image path (default: denoised.png)
-g, --gt <path>         #Clean ground truth image path for metric computation (default: TEST_GT.png)
-k, --kernel-size <N>   #Kernel size (must be a positive odd integer, default: 15)
-p, --passes <N>        #Number of A Trous passes (default: 5)
-sc, --sigma-color <V>  #Color sigma parameter (default: 0.15)
-sn, --sigma-normal <V> #Normal sigma parameter (default: 0.1)
-sa, --sigma-albedo <V> #Albedo sigma parameter (default: 0.05)
-mk, --median-kernel <N>   #Median kernel size (3 or 5, default: 3)
-mt, --median-threshold <V> #Median outlier threshold (default: 0.05)
-r, --runs <N>          #Number of iterations to run during benchmark mode (default: 5)
```

### Running a Specific Filter

Ex- To run the CUDA A Trous denoiser on custom images:

```bash
./build/bigdn -f atrous -i input.png -n normal.png -a albedo.png -o output.png -p 5 -sc 0.15 -sn 0.1
```

## Denoising Metrics

Below is an example of the quality metrics obtained relative to the clean reference image. Higher values for both PSNR and MS-SSIM indicate better denoising quality:

![Denoising Interface Screenshot](gui_screenshot.png)

* Mean Filter: 28.23 dB PSNR, 0.9807 MS-SSIM
* Gaussian Filter: 34.49 dB PSNR, 0.9885 MS-SSIM
* Guided Filter: 28.20 dB PSNR, 0.9898 MS-SSIM
* Joint Guided Filter 5x5 kernel: 35.7724 dB PSNR, 0.9989 MS-SSIM 
* A Trous Wavelet Filter (3, 0.18, 0.15): 33.6352 dB PSNR, 0.9985 MS-SSIM
* Masked Median + A-Trous Wavelet Hybrid Filter (3, 0.84, 0.07, 5, 0.015): 33.42 dB PSNR, 0.9981 MS-SSIM

## Known Issues

* Classical edge preserving filters (Guided, Joint Guided, A Trous) have a hard mathematical limit distinguishing high frequency noise from true details, which can leave splotchiness in high noise renders.
* The Joint Guided and A Trous filters are only supported on the CUDA device(blame my laziness).
* Standard A Trous does not denoise salt and pepper noise well because pixel outlier spikes are treated as edges and are not smoothed. Somewhat resolved by using the **Masked Median + A-Trous Wavelet Filter**, which pre-filters outlier spike pixels with a selective median filter before using variance-guided A Trous denoising.
* Joint Guided filter is prone to smudginess and detail loss.
* Gaussian filter is too smudgy and completely blurs geometric edges.
