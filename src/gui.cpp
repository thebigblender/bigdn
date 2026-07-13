#include "gui.hpp"
#include "image.hpp"
#include "filters.hpp"
#include "metrics.hpp"
#include <nanobench.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

// OpenGL texture IDs
static GLuint g_inputTex = 0;
static GLuint g_denoisedTex = 0;

static Image g_inputImg;
static Image g_normalImg;
static Image g_albedoImg;
static Image g_gtImg;
static Image g_denoisedImg;

static bool g_inputLoaded = false;
static bool g_normalLoaded = false;
static bool g_albedoLoaded = false;
static bool g_gtLoaded = false;

// UI Paths
static char g_inputPath[512] = "TEST.png";
static char g_normalPath[512] = "TEST_NORMAL.png";
static char g_albedoPath[512] = "TEST_ALBEDO.png";
static char g_gtPath[512] = "TEST_GT.png";

static int g_filterIdx = 4; // default to a-trous
static const char* g_filters[] = { "Mean", "Gaussian", "Guided", "Joint Guided", "A-Trous", "Masked Median + A-Trous" };

static int g_medianKernelSize = 3;
static float g_medianThreshold = 0.05f;

static int g_deviceIdx = 2; // default to cuda
static const char* g_devices[] = { "CPU", "OpenMP", "CUDA" };

static int g_kernelSize = 15;
static int g_passes = 5;
static float g_sigmaColor = 0.15f;
static float g_sigmaNormal = 0.1f;
static float g_sigmaAlbedo = 0.05f;
static float g_eps = 0.005f;

static float g_splitFraction = 0.5f;

// Benchmark data
static bool g_runBenchmarkTriggered = false;
static std::vector<std::string> g_benchNames;
static std::vector<std::string> g_benchTimes;
static std::vector<std::string> g_benchErrors;
static std::vector<std::string> g_benchSpeedups;
static std::vector<std::string> g_qualityPsnrs;
static std::vector<std::string> g_qualityMsssims;

// Metrics cache
static float g_cachedPsnr = 0.0f;
static float g_cachedMsssim = 0.0f;

// Image placeholders
static Image makePlaceholder(int w, int h, float val) {
    Image img(w, h, 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            img.setPixel(x, y, 0, val);
            img.setPixel(x, y, 1, val);
            img.setPixel(x, y, 2, val);
        }
    }
    return img;
}

// Upload pixels to OpenGL texture
static void uploadToTexture(GLuint textureId, const Image& img) {
    glBindTexture(GL_TEXTURE_2D, textureId);
    
    // set parameters to make texture complete and prevent black screen rendering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    int w = img.getWidth();
    int h = img.getHeight();
    int c = img.getChannels();
    std::vector<unsigned char> data(w * h * c);
    
    int plane = w * h;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int destIdx = (y * w + x) * c;
            int srcIdx = y * w + x;
            if (c == 3) {
                float r = img.getData()[srcIdx];
                float g = img.getData()[plane + srcIdx];
                float b = img.getData()[2 * plane + srcIdx];
                data[destIdx]     = static_cast<unsigned char>(std::max(0.0f, std::min(r, 1.0f)) * 255.0f);
                data[destIdx + 1] = static_cast<unsigned char>(std::max(0.0f, std::min(g, 1.0f)) * 255.0f);
                data[destIdx + 2] = static_cast<unsigned char>(std::max(0.0f, std::min(b, 1.0f)) * 255.0f);
            } else {
                float v = img.getData()[srcIdx];
                data[destIdx] = static_cast<unsigned char>(std::max(0.0f, std::min(v, 1.0f)) * 255.0f);
            }
        }
    }
    
    GLenum format = (c == 3) ? GL_RGB : GL_RED;
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data.data());
}

static void loadTextures() {
    g_inputLoaded = g_inputImg.load(g_inputPath);
    if (!g_inputLoaded) {
        g_inputImg = makePlaceholder(512, 512, 0.2f);
    }
    uploadToTexture(g_inputTex, g_inputImg);

    g_normalLoaded = g_normalImg.load(g_normalPath);
    if (!g_normalLoaded) {
        g_normalImg = makePlaceholder(g_inputImg.getWidth(), g_inputImg.getHeight(), 0.5f);
    }

    g_albedoLoaded = g_albedoImg.load(g_albedoPath);
    if (!g_albedoLoaded) {
        g_albedoImg = makePlaceholder(g_inputImg.getWidth(), g_inputImg.getHeight(), 0.8f);
    }

    g_gtLoaded = g_gtImg.load(g_gtPath);
    if (!g_gtLoaded) {
        g_gtImg = g_inputImg;
    }
}

static void runFiltering() {
    if (!g_inputLoaded) return;
    
    try {
        if (g_filterIdx == 0) { // Mean
            if (g_deviceIdx == 0) {
                g_denoisedImg = Filters::meanCPU_ST(g_inputImg, g_kernelSize);
            } else if (g_deviceIdx == 1) {
                g_denoisedImg = Filters::meanCPU_OMP(g_inputImg, g_kernelSize);
            } else {
                g_denoisedImg = Filters::meanCUDA(g_inputImg, g_kernelSize);
            }
        } else if (g_filterIdx == 1) { // Gaussian
            if (g_deviceIdx == 0) {
                g_denoisedImg = Filters::gaussianCPU_ST(g_inputImg, g_kernelSize, 1.5f);
            } else if (g_deviceIdx == 1) {
                g_denoisedImg = Filters::gaussianCPU_OMP(g_inputImg, g_kernelSize, 1.5f);
            } else {
                g_denoisedImg = Filters::gaussianCUDA(g_inputImg, g_kernelSize, 1.5f);
            }
        } else if (g_filterIdx == 2) { // Guided
            if (g_deviceIdx == 0) {
                g_denoisedImg = Filters::guidedCPU_ST(g_inputImg, g_normalImg, g_kernelSize, g_eps);
            } else if (g_deviceIdx == 1) {
                g_denoisedImg = Filters::guidedCPU_OMP(g_inputImg, g_normalImg, g_kernelSize, g_eps);
            } else {
                g_denoisedImg = Filters::guidedCUDA(g_inputImg, g_normalImg, g_kernelSize, g_eps);
            }
        } else if (g_filterIdx == 3) { // Joint Guided
            g_denoisedImg = Filters::jointGuidedCUDA(g_inputImg, g_normalImg, g_albedoImg, g_kernelSize, g_eps);
        } else if (g_filterIdx == 4) { // A-Trous
            g_denoisedImg = Filters::aTrousWaveletCUDA(g_inputImg, g_normalImg, g_albedoImg, g_passes, g_sigmaColor, g_sigmaNormal, g_sigmaAlbedo);
        } else if (g_filterIdx == 5) { // Masked Median + A-Trous
            g_denoisedImg = Filters::maskedMedianAtrousCUDA(g_inputImg, g_normalImg, g_albedoImg, g_passes, g_sigmaColor, g_sigmaNormal, g_sigmaAlbedo, g_medianKernelSize, g_medianThreshold);
        }

        if (g_gtLoaded) {
            g_cachedPsnr = Metrics::calculatePSNR(g_gtImg, g_denoisedImg);
            g_cachedMsssim = Metrics::calculateMSSIM(g_gtImg, g_denoisedImg);
        } else {
            g_cachedPsnr = 0.0f;
            g_cachedMsssim = 0.0f;
        }
        
        uploadToTexture(g_denoisedTex, g_denoisedImg);
    } catch (const std::exception& e) {
        std::cerr << "Filter exception: " << e.what() << std::endl;
    }
}

static std::string formatTime(double sec) {
    double ms = sec * 1000.0;
    char buf[64];
    if (ms >= 1.0) {
        std::snprintf(buf, sizeof(buf), "%.3f ms", ms);
    } else {
        std::snprintf(buf, sizeof(buf), "%.3f us", ms * 1000.0);
    }
    return std::string(buf);
}

static std::string formatDouble(double val) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f", val);
    return std::string(buf);
}

static std::string formatDouble4(double val) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4f", val);
    return std::string(buf);
}

static void addBenchResult(const std::string& name, double t, double err, double speedup, const Image& out) {
    g_benchNames.push_back(name);
    g_benchTimes.push_back(formatTime(t));
    g_benchErrors.push_back(formatDouble(err) + "%");
    g_benchSpeedups.push_back(formatDouble(speedup) + "x");
    if (g_gtLoaded) {
        g_qualityPsnrs.push_back(formatDouble(Metrics::calculatePSNR(g_gtImg, out)));
        g_qualityMsssims.push_back(formatDouble4(Metrics::calculateMSSIM(g_gtImg, out)));
    } else {
        g_qualityPsnrs.push_back("N/A");
        g_qualityMsssims.push_back("N/A");
    }
}

namespace GUI {

void launch() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(1440, 900, "bigdn - Real-time Denoising Toolkit", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    glGenTextures(1, &g_inputTex);
    glGenTextures(1, &g_denoisedTex);

    loadTextures();
    runFiltering();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2((float)w, (float)h));
        ImGui::Begin("Main Window", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        // Sidebar Panel
        ImGui::BeginChild("ControlPanel", ImVec2(380, 0), true);
        
        ImGui::Text("File Paths");
        ImGui::InputText("Noisy Input", g_inputPath, sizeof(g_inputPath));
        ImGui::InputText("Normal Guide", g_normalPath, sizeof(g_normalPath));
        ImGui::InputText("Albedo Guide", g_albedoPath, sizeof(g_albedoPath));
        ImGui::InputText("Ground Truth", g_gtPath, sizeof(g_gtPath));

        if (ImGui::Button("Load Defaults")) {
            std::strcpy(g_inputPath, "TEST.png");
            std::strcpy(g_normalPath, "TEST_NORMAL.png");
            std::strcpy(g_albedoPath, "TEST_ALBEDO.png");
            std::strcpy(g_gtPath, "TEST_GT.png");
            loadTextures();
            runFiltering();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload Images")) {
            loadTextures();
            runFiltering();
        }

        ImGui::Separator();
        ImGui::Text("Denoising Configuration");
        if (ImGui::Combo("Filter", &g_filterIdx, g_filters, IM_ARRAYSIZE(g_filters))) {
            runFiltering();
        }
        
        if (g_filterIdx < 3) {
            if (ImGui::Combo("Device", &g_deviceIdx, g_devices, IM_ARRAYSIZE(g_devices))) {
                runFiltering();
            }
        } else {
            ImGui::Text("Device: CUDA (GPU Only)");
        }

        ImGui::Separator();
        bool paramsChanged = false;

        if (g_filterIdx == 0 || g_filterIdx == 1 || g_filterIdx == 2 || g_filterIdx == 3) {
            int oldK = g_kernelSize;
            if (ImGui::SliderInt("Kernel Size", &g_kernelSize, 3, 31)) {
                if (g_kernelSize % 2 == 0) g_kernelSize++;
                if (g_kernelSize != oldK) paramsChanged = true;
            }
        }

        if (g_filterIdx == 1) {
            if (ImGui::SliderFloat("Sigma Color", &g_sigmaColor, 0.1f, 10.0f, "%.2f")) {
                paramsChanged = true;
            }
        }

        if (g_filterIdx == 2 || g_filterIdx == 3) {
            if (ImGui::SliderFloat("Epsilon (eps)", &g_eps, 0.0001f, 0.1f, "%.4f")) {
                paramsChanged = true;
            }
        }

        if (g_filterIdx == 4 || g_filterIdx == 5) {
            if (ImGui::SliderInt("Passes", &g_passes, 1, 10)) {
                paramsChanged = true;
            }
            if (ImGui::SliderFloat("Sigma Color", &g_sigmaColor, 0.01f, 1.0f, "%.2f")) {
                paramsChanged = true;
            }
            if (ImGui::SliderFloat("Sigma Normal", &g_sigmaNormal, 0.01f, 1.0f, "%.2f")) {
                paramsChanged = true;
            }
        }

        if (g_filterIdx == 5) {
            int oldMK = g_medianKernelSize;
            if (ImGui::SliderInt("Median Kernel Size", &g_medianKernelSize, 3, 5)) {
                if (g_medianKernelSize % 2 == 0) g_medianKernelSize = (g_medianKernelSize < 4) ? 3 : 5;
                if (g_medianKernelSize != oldMK) paramsChanged = true;
            }
            if (ImGui::SliderFloat("Median Threshold", &g_medianThreshold, 0.001f, 0.5f, "%.3f")) {
                paramsChanged = true;
            }
        }

        if (paramsChanged) {
            runFiltering();
        }

        ImGui::Separator();
        ImGui::Text("Denoising Metrics");
        if (g_gtLoaded && g_inputLoaded) {
            ImGui::Text("PSNR: %.2f dB", g_cachedPsnr);
            ImGui::Text("MS-SSIM: %.4f", g_cachedMsssim);
        } else {
            ImGui::Text("No ground truth loaded for metrics.");
        }

        ImGui::Separator();
        if (ImGui::Button("Run Benchmark Suite", ImVec2(-1, 0))) {
            g_runBenchmarkTriggered = true;
        }

        ImGui::EndChild();

        ImGui::SameLine();

        // Viewport & Benchmark Tabs
        ImGui::BeginChild("MainArea", ImVec2(0, 0), true);

        if (g_runBenchmarkTriggered) {
            g_benchNames.clear();
            g_benchTimes.clear();
            g_benchErrors.clear();
            g_benchSpeedups.clear();
            g_qualityPsnrs.clear();
            g_qualityMsssims.clear();

            double meanStTime = 1.0;
            // Mean ST
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Mean ST", [&] { out = Filters::meanCPU_ST(g_inputImg, g_kernelSize); });
                meanStTime = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Mean CPU ST", meanStTime, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, 1.0, out);
            }
            // Mean CUDA
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Mean CUDA", [&] { out = Filters::meanCUDA(g_inputImg, g_kernelSize); });
                double t = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Mean CUDA", t, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, meanStTime / t, out);
            }
            // Gaussian ST
            double gaussStTime = 1.0;
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Gaussian ST", [&] { out = Filters::gaussianCPU_ST(g_inputImg, g_kernelSize, 1.5f); });
                gaussStTime = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Gaussian CPU ST", gaussStTime, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, 1.0, out);
            }
            // Gaussian CUDA
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Gaussian CUDA", [&] { out = Filters::gaussianCUDA(g_inputImg, g_kernelSize, 1.5f); });
                double t = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Gaussian CUDA", t, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, gaussStTime / t, out);
            }
            // Guided ST
            double guidedStTime = 1.0;
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Guided ST", [&] { out = Filters::guidedCPU_ST(g_inputImg, g_normalImg, g_kernelSize, g_eps); });
                guidedStTime = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Guided CPU ST", guidedStTime, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, 1.0, out);
            }
            // Guided CUDA
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Guided CUDA", [&] { out = Filters::guidedCUDA(g_inputImg, g_normalImg, g_kernelSize, g_eps); });
                double t = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Guided CUDA", t, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, guidedStTime / t, out);
            }
            // Joint Guided CUDA
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Joint Guided CUDA", [&] { out = Filters::jointGuidedCUDA(g_inputImg, g_normalImg, g_albedoImg, g_kernelSize, g_eps); });
                double t = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Joint Guided CUDA", t, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, guidedStTime / t, out);
            }
            // A-Trous CUDA
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("A-Trous CUDA", [&] { out = Filters::aTrousWaveletCUDA(g_inputImg, g_normalImg, g_albedoImg, g_passes, g_sigmaColor, g_sigmaNormal, g_sigmaAlbedo); });
                double t = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("A-Trous CUDA", t, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, guidedStTime / t, out);
            }
            // Masked Median + A-Trous CUDA
            {
                ankerl::nanobench::Bench bench;
                bench.epochs(3).minEpochTime(std::chrono::milliseconds(20)).output(nullptr);
                Image out;
                bench.run("Masked Median + A-Trous CUDA", [&] { out = Filters::maskedMedianAtrousCUDA(g_inputImg, g_normalImg, g_albedoImg, g_passes, g_sigmaColor, g_sigmaNormal, g_sigmaAlbedo, g_medianKernelSize, g_medianThreshold); });
                double t = bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
                addBenchResult("Masked Median + A-Trous CUDA", t, bench.results().front().medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0, guidedStTime / t, out);
            }

            g_runBenchmarkTriggered = false;
        }

        if (ImGui::BeginTabBar("MainTabBar")) {
            if (ImGui::BeginTabItem("Split Viewport")) {
                ImVec2 avail = ImGui::GetContentRegionAvail();
                float imgW = (float)g_inputImg.getWidth();
                float imgH = (float)g_inputImg.getHeight();
                
                if (imgW > 0 && imgH > 0) {
                    float aspect = imgW / imgH;
                    float viewW = avail.x;
                    float viewH = avail.x / aspect;
                    if (viewH > avail.y) {
                        viewH = avail.y;
                        viewW = avail.y * aspect;
                    }

                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail.x - viewW) * 0.5f);
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail.y - viewH) * 0.5f);

                    ImVec2 screenPos = ImGui::GetCursorScreenPos();
                    ImVec2 endPos = ImVec2(screenPos.x + viewW, screenPos.y + viewH);

                    ImDrawList* drawList = ImGui::GetWindowDrawList();
                    
                    // draw noisy image
                    drawList->AddImage((ImTextureID)(intptr_t)g_inputTex, screenPos, endPos);

                    // split coordinate
                    float splitX = screenPos.x + g_splitFraction * viewW;
                    
                    // draw denoised image clipped to right side
                    drawList->PushClipRect(ImVec2(splitX, screenPos.y), endPos, true);
                    drawList->AddImage((ImTextureID)(intptr_t)g_denoisedTex, screenPos, endPos);
                    drawList->PopClipRect();

                    // draw line separator
                    drawList->AddLine(ImVec2(splitX, screenPos.y), ImVec2(splitX, endPos.y), IM_COL32(255, 255, 255, 255), 3.0f);

                    // handle split interaction
                    ImGui::SetCursorScreenPos(screenPos);
                    ImGui::InvisibleButton("##split_slider", ImVec2(viewW, viewH));
                    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                        float mx = ImGui::GetIO().MousePos.x;
                        g_splitFraction = (mx - screenPos.x) / viewW;
                        g_splitFraction = std::max(0.0f, std::min(g_splitFraction, 1.0f));
                    }
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Benchmark Results")) {
                if (g_benchNames.empty()) {
                    ImGui::Text("Click 'Run Benchmark Suite' in the sidebar to run the benchmarks.");
                } else {
                    if (ImGui::BeginTable("BenchmarkTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Algorithm");
                        ImGui::TableSetupColumn("Median Time");
                        ImGui::TableSetupColumn("Error %");
                        ImGui::TableSetupColumn("Speedup");
                        ImGui::TableSetupColumn("PSNR (dB)");
                        ImGui::TableSetupColumn("MS-SSIM");
                        ImGui::TableHeadersRow();

                        for (size_t i = 0; i < g_benchNames.size(); ++i) {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0); ImGui::Text("%s", g_benchNames[i].c_str());
                            ImGui::TableSetColumnIndex(1); ImGui::Text("%s", g_benchTimes[i].c_str());
                            ImGui::TableSetColumnIndex(2); ImGui::Text("%s", g_benchErrors[i].c_str());
                            ImGui::TableSetColumnIndex(3); ImGui::Text("%s", g_benchSpeedups[i].c_str());
                            ImGui::TableSetColumnIndex(4); ImGui::Text("%s", g_qualityPsnrs[i].c_str());
                            ImGui::TableSetColumnIndex(5); ImGui::Text("%s", g_qualityMsssims[i].c_str());
                        }
                        ImGui::EndTable();
                    }
                }
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        ImGui::EndChild();

        ImGui::End();

        // Rendering
        ImGui::Render();
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // cleanup
    glDeleteTextures(1, &g_inputTex);
    glDeleteTextures(1, &g_denoisedTex);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

} // namespace GUI
