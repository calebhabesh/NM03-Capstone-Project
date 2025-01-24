#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Function to perform contrast stretching on the image
// Contrast stretching enhances the contrast by scaling the pixel intensity
// range to fit a desired range, here it's set to 0-255.
std::vector<std::vector<int>> contrastStretching(const std::vector<std::vector<int>>& image, int newMin, int newMax) {
    int rows = image.size();  // Get the number of rows in the image
    int cols = image[0].size();  // Get the number of columns in the image

    // Initialize oldMin and oldMax to find the minimum and maximum pixel values in the image
    int oldMin = 255, oldMax = 0;

    // Loop through each pixel in the image to find oldMin and oldMax
    for (const auto& row : image) {
        for (int pixel : row) {
            oldMin = std::min(oldMin, pixel);  // Update oldMin with the smallest pixel value
            oldMax = std::max(oldMax, pixel);  // Update oldMax with the largest pixel value
        }
    }

    // Create an empty 2D vector (same size as the input image) to store the stretched image
    std::vector<std::vector<int>> stretchedImage(rows, std::vector<int>(cols, 0));

    // Apply contrast stretching to each pixel
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Scale the pixel intensity to the new range [newMin, newMax]
            stretchedImage[i][j] = (image[i][j] - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;
        }
    }

    // Return the contrast-stretched image
    return stretchedImage;
}

// Function to apply Gaussian blur for noise reduction
// This function uses a 3x3 Gaussian kernel for smoothing the image, reducing high-frequency noise.
std::vector<std::vector<int>> gaussianFilter(const std::vector<std::vector<int>>& image) {
    // Define a 3x3 Gaussian kernel (with values that give a basic smoothing effect)
    std::vector<std::vector<float>> kernel = {
        {1 / 16.0, 2 / 16.0, 1 / 16.0},
        {2 / 16.0, 4 / 16.0, 2 / 16.0},
        {1 / 16.0, 2 / 16.0, 1 / 16.0}
    };

    int rows = image.size();  // Get the number of rows in the image
    int cols = image[0].size();  // Get the number of columns in the image
    int kernelSize = kernel.size();  // Kernel size (3x3)
    int offset = kernelSize / 2;  // Offset for the kernel (1 for a 3x3 kernel)

    // Create an empty 2D vector to store the filtered image
    std::vector<std::vector<int>> filteredImage(rows, std::vector<int>(cols, 0));

    // Apply the Gaussian filter by convolving the kernel with each pixel in the image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0.0;  // Initialize a variable to accumulate the weighted sum of neighbors

            // Loop through the kernel and apply it to the neighboring pixels
            for (int ki = -offset; ki <= offset; ++ki) {
                for (int kj = -offset; kj <= offset; ++kj) {
                    int ni = i + ki;  // Neighbor row index
                    int nj = j + kj;  // Neighbor column index

                    // Skip neighbors that are out of bounds (outside the image)
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        // Apply the kernel to the pixel
                        sum += image[ni][nj] * kernel[ki + offset][kj + offset];
                    }
                }
            }

            // Store the result in the filtered image (convert sum to an integer)
            filteredImage[i][j] = static_cast<int>(sum);
        }
    }

    // Return the filtered (smoothed) image
    return filteredImage;
}

int main() {
    // Define an example 5x5 grayscale image
    // Pixel values represent grayscale intensities in the range 0–255
    std::vector<std::vector<int>> image = {
        {50, 80, 90, 100, 60},
        {70, 120, 150, 130, 80},
        {90, 140, 200, 160, 100},
        {80, 130, 170, 150, 90},
        {60, 100, 110, 120, 70}
    };

    // Step 1: Apply contrast stretching to enhance the contrast of the image
    std::vector<std::vector<int>> enhancedImage = contrastStretching(image, 0, 255);

    // Step 2: Apply Gaussian filter to reduce noise
    std::vector<std::vector<int>> filteredImage = gaussianFilter(enhancedImage);

    // Output the contrast-enhanced image
    std::cout << "Enhanced Image:" << std::endl;
    for (const auto& row : enhancedImage) {
        for (const auto& pixel : row) {
            std::cout << pixel << " ";  // Print each pixel value
        }
        std::cout << std::endl;
    }

    std::cout << "\nFiltered Image:" << std::endl;
    // Output the filtered image after Gaussian smoothing
    for (const auto& row : filteredImage) {
        for (const auto& pixel : row) {
            std::cout << pixel << " ";  // Print each pixel value
        }
        std::cout << std::endl;
    }

    return 0;  
}
