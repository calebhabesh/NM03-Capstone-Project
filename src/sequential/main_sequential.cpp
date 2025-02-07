// FAST framework libs
#include <FAST/Algorithms/GaussianSmoothing/GaussianSmoothing.hpp>
#include <FAST/Algorithms/ImageCaster/ImageCaster.hpp>
#include <FAST/Algorithms/ImageSharpening/ImageSharpening.hpp>
#include <FAST/Algorithms/IntensityClipping/IntensityClipping.hpp>
#include <FAST/Algorithms/IntensityNormalization/IntensityNormalization.hpp>
#include <FAST/Algorithms/VectorMedianFilter/VectorMedianFilter.hpp>
#include <FAST/Config.hpp>
#include <FAST/Data/Image.hpp>
#include <FAST/Importers/DICOMFileImporter.hpp>
#include <FAST/Pipeline.hpp>

// cpp libs
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace fast;
namespace fs = std::filesystem;

class SequentialBrainTumorProcessor {
private:
  std::vector<std::string> dicomFiles;
  std::string basePath;

  // Timing measurements
  std::chrono::duration<double> importTime{0};
  std::chrono::duration<double> preprocessTime{0};
  std::chrono::duration<double> totalTime{0};

  // Helper function to extract number from filename
  int extractFileNumber(const std::string &filename) {
    // Extract the number between the last '-' and '.dcm'
    size_t dashPos = filename.find_last_of('-');
    size_t dotPos = filename.find(".dcm");
    if (dashPos != std::string::npos && dotPos != std::string::npos) {
      std::string numStr = filename.substr(dashPos + 1, dotPos - dashPos - 1);
      try {
        return std::stoi(numStr);
      } catch (...) {
        return 1000; // Return large number for invalid cases
      }
    }
    return 1000;
  }

public:
  SequentialBrainTumorProcessor() {
    basePath = Config::getTestDataPath() +
               "Brain-Tumor-Progression/PGBM-017/09-17-1997-RA FH MR RCBV "
               "OP-85753/16.000000-T1post-19554/";
    loadDICOMFiles();
  }

  void loadDICOMFiles() {
    try {
      std::vector<std::pair<std::string, int>> fileNumberPairs;

      // Collect files and their numbers
      for (const auto &entry : fs::directory_iterator(basePath)) {
        if (entry.path().extension() == ".dcm") {
          std::string filepath = entry.path().string();
          int fileNumber = extractFileNumber(entry.path().filename().string());
          fileNumberPairs.push_back({filepath, fileNumber});
        }
      }

      // Sort based on file numbers
      std::sort(
          fileNumberPairs.begin(), fileNumberPairs.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });

      // Extract sorted file paths
      dicomFiles.clear();
      for (const auto &pair : fileNumberPairs) {
        dicomFiles.push_back(pair.first);
      }

      std::cout << "Found " << dicomFiles.size()
                << " DICOM files in: " << basePath << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error loading DICOM files: " << e.what() << std::endl;
    }
  }
  void processSingleImage(const std::string &filename) {
    auto startTotal = std::chrono::high_resolution_clock::now();

    try {
      // Import DICOM
      auto startImport = std::chrono::high_resolution_clock::now();
      auto importer = DICOMFileImporter::create(filename);

      importer->setLoadSeries(false); // Load only one 2D slice

      auto importedImage = importer->runAndGetOutputData<Image>();
      importTime += std::chrono::high_resolution_clock::now() - startImport;

      // Preprocessing steps
      auto startPreprocess = std::chrono::high_resolution_clock::now();

      // 1. Intensity Normalization
      auto normalize =
          IntensityNormalization::create(0.5f, 2.5f, 0.0f, 10000.0f);
      normalize->connect(importer);

      // 2. Intensity Clipping
      auto clipping = IntensityClipping::create(0.65f, 4000.0f);
      clipping->connect(normalize);

      // 3. Vector Median Filter (Denoise and preserve edges)
      auto medianfilter = VectorMedianFilter::create(5);
      medianfilter->connect(clipping);

      // 4. Image Sharpening
      auto sharpen = ImageSharpening::create(2.0f, 0.5f, 9);
      sharpen->connect(medianfilter);

      // Get the final processed image
      auto processedImage = sharpen->runAndGetOutputData<Image>();

      preprocessTime +=
          std::chrono::high_resolution_clock::now() - startPreprocess;

      // Here you can add code to save or further process the image if needed

    } catch (const std::exception &e) {
      std::cerr << "Error processing file " << filename << ": " << e.what()
                << std::endl;
    }

    totalTime += std::chrono::high_resolution_clock::now() - startTotal;
  }

  void processAllImages() {
    std::cout << "Starting sequential processing of " << dicomFiles.size()
              << " images from directory: " << basePath << std::endl;

    for (const auto &file : dicomFiles) {
      std::cout << "Processing: " << fs::path(file).filename().string()
                << std::endl;
      processSingleImage(file);
    }

    printTimingResults();
  }

  void printTimingResults() const {
    std::cout << "\nProcessing Time Results:" << std::endl;
    std::cout << "Import Time: " << importTime.count() << " seconds"
              << std::endl;
    std::cout << "Preprocessing Time: " << preprocessTime.count() << " seconds"
              << std::endl;
    std::cout << "Total Time: " << totalTime.count() << " seconds" << std::endl;
    std::cout << "Average Time per Image: "
              << totalTime.count() / dicomFiles.size() << " seconds"
              << std::endl;
  }
};

int main() {
  try {
    SequentialBrainTumorProcessor processor;
    processor.processAllImages();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
