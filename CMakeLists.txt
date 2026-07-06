cmake_minimum_required(VERSION 3.18)
project(DenoiserToolkit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check for CUDA compiler optionally
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    endif()
    add_compile_definitions(WITH_CUDA)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    set(CUDA_ENABLED FALSE)
    message(WARNING "CUDA compiler (nvcc) not found. GPU filters will be compiled as CPU-only or stubbed out.")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)

# Source files (we list them explicitly to avoid GLOB caching issues)
set(SOURCES
    src/main.cpp
    src/benchmark.cpp
    src/image.cpp
    src/metrics.cpp
    
    # Classical Filters (Uncomment and implement these files as you create them)
    src/filters/mean_ST.cpp
    src/filters/mean_OMP.cpp
    # src/filters/gaussian.cpp
    # src/filters/median.cpp
    # src/filters/bilateral.cpp
    # src/filters/guided.cpp
    # src/filters/nlm.cpp
    
    # Learned Filters
    # src/filters/dncnn.cpp
)

# Add CUDA source files only if CUDA was found
if(CUDA_ENABLED)
    list(APPEND SOURCES
        src/filters/mean_CUDA.cu
    )
endif()

# Executable target
add_executable(denoiser_toolkit ${SOURCES})

# Target properties & linking
target_link_libraries(denoiser_toolkit
    PRIVATE
        OpenMP::OpenMP_CXX
)

# Optional: Enable max warnings for quality
if(MSVC)
    target_compile_options(denoiser_toolkit PRIVATE /W4)
else()
    target_compile_options(denoiser_toolkit PRIVATE -Wall -Wextra -O3)
endif()
