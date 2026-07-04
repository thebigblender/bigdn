#include "../include/image.hpp"

#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image()
    : width(0), height(0), channels(0)
{}

Image::Image(int w, int h, int c)
    : width(w), height(h), channels(c), data(w * h * c)
{}

float Image::getPixel(int x, int y, int c) const {
    return data[index(x, y, c)];
}

float Image::getPixelClamped(int x, int y, int c) const {
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    c = std::max(0, std::min(c, channels - 1));
    return getPixel(x, y, c);
}

void Image::setPixel(int x, int y, int c, float value)
{
    data[index(x, y, c)] = value;
}

bool Image::load(const std::string& path)
{
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!pixels) return false;
    channels = 3;

    data.resize(width * height * channels);

    //transpose interleaved HWC to planar CHW
    for (int c = 0; c < channels; c++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int interleaved =
                    (y * width + x) * channels + c;

                int planar =
                    index(x, y, c);

                data[planar] =
                    pixels[interleaved] / 255.0f;
            }
        }
    }

    stbi_image_free(pixels);

    return true;
}

bool Image::save(const std::string& path) const
{
    if (width <= 0 || height <= 0 || channels <= 0 || data.empty()) {
        return false;
    }

    std::vector<unsigned char> pixels(width * height * channels);

    // Reverse transpose planar CHW to interleaved HWC and denormalize
    for (int c = 0; c < channels; c++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int planar = index(x, y, c);
                int interleaved = (y * width + x) * channels + c;

                float val = data[planar] * 255.0f;
                pixels[interleaved] = static_cast<unsigned char>(
                    std::min(std::max(val, 0.0f), 255.0f)
                );
            }
        }
    }

    // Determine format by extension
    size_t dot_pos = path.find_last_of(".");
    std::string ext = (dot_pos == std::string::npos) ? "" : path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    int success = 0;
    if (ext == "png") {
        success = stbi_write_png(path.c_str(), width, height, channels, pixels.data(), width * channels);
    } else if (ext == "jpg" || ext == "jpeg") {
        success = stbi_write_jpg(path.c_str(), width, height, channels, pixels.data(), 90);
    } else if (ext == "bmp") {
        success = stbi_write_bmp(path.c_str(), width, height, channels, pixels.data());
    } else {
        // default to png
        success = stbi_write_png(path.c_str(), width, height, channels, pixels.data(), width * channels);
    }

    return success != 0;
}