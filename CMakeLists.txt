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
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    set(CUDA_ENABLED FALSE)
    message(WARNING "CUDA compiler (nvcc) not found. GPU filters will be compiled as CPU-only or stubbed out.")
endif()

# Find GLFW and OpenGL
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# Fetch Dear ImGui
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90
)
FetchContent_MakeAvailable(imgui)

# Set ImGui directories and sources
set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

# Source files (we list them explicitly to avoid GLOB caching issues)
set(SOURCES
    src/main.cpp
    src/benchmark.cpp
    src/image.cpp
    src/metrics.cpp
    src/gui.cpp
    
    # Classical Filters (Uncomment and implement these files as you create them)
    src/filters/mean_ST.cpp
    src/filters/mean_OMP.cpp
    src/filters/gaussian_ST.cpp
    src/filters/gaussian_OMP.cpp
    # src/filters/median.cpp
    # src/filters/bilateral.cpp
    src/filters/guided_ST.cpp
    src/filters/guided_OMP.cpp
    # src/filters/nlm.cpp
    
    # Learned Filters
    # src/filters/dncnn.cpp
)

# Add CUDA source files only if CUDA was found
if(CUDA_ENABLED)
    list(APPEND SOURCES
        src/filters/mean_CUDA.cu
        src/filters/gaussian_CUDA.cu
        src/filters/guided_CUDA.cu
        src/filters/joint_guided_CUDA.cu
        src/filters/ATrousWavelet_CUDA.cu
    )
endif()

# Append ImGui source files
list(APPEND SOURCES ${IMGUI_SOURCES})

# Executable target
add_executable(bigdn ${SOURCES})

# Target properties & linking
target_link_libraries(bigdn
    PRIVATE
        OpenMP::OpenMP_CXX
        glfw
        OpenGL::GL
        ${CMAKE_DL_LIBS}
)

if(CUDA_ENABLED)
    target_link_libraries(bigdn
        PRIVATE
            CUDA::cudart
    )
endif()

# Optional: Enable max warnings for quality
if(MSVC)
    target_compile_options(bigdn PRIVATE /W4)
else()
    target_compile_options(bigdn PRIVATE -Wall -Wextra -O3)
endif()
