#pragma once
#include <string>
#include <vector>

class Image {
public:
    Image();
    Image(int width, int height, int channels = 3);

    bool load(const std::string& path);
    bool save(const std::string& path) const;

    float getPixel(int x, int y, int c) const;
    float getPixelClamped(int x, int y, int c) const;
    
    void setPixel(int x, int y, int c, float value);
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }

    float* getData() { return data.data(); }
    const float* getData() const { return data.data(); }

private:
    int width;
    int height;
    int channels;

    std::vector<float> data;

    inline int index (int x, int y, int c) const {
        return c * width * height + y * width + x;
    }
};

