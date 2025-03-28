#include "FAST/includes.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

using namespace fast;
namespace fs = std::filesystem;

class SequentialImageProcessor {
private:
  std::vector<std::string> dicomFiles;
  std::string basePath;
  std::string outputPath;

  // Helper function to extract number from filename for sorting
  int extractFileNumber(const std::string &filename) {
    size_t dashPos = filename.find_last_of('-');
    size_t dotPos = filename.find(".dcm");
    if (dashPos != std::string::npos && dotPos != std::string::npos) {
      std::string numStr = filename.substr(dashPos + 1, dotPos - dashPos - 1);
      try {
        return std::stoi(numStr);
      } catch (...) {
        return 1000;
      }
    }
    return 1000;
  }

  void setupOutputDirectory() {
    try {
      if (system(("mkdir -p " + outputPath + " && cd " + outputPath +
                  " && rm -rf *")
                     .c_str()) != 0) {
        throw std::runtime_error("Failed to setup output directory: " +
                                 outputPath);
      }
      std::cout << "Created clean output directory: " + outputPath << std::endl;
    } catch (const std::exception &e) {
      throw std::runtime_error("Error setting up output directory: " +
                               std::string(e.what()));
    }
  }

  void
  exportProcessedImage(const std::string &filename,
                       std::shared_ptr<RenderToImage> renderToImage,
                       std::shared_ptr<ImageRenderer> originalRenderer,
                       std::shared_ptr<SegmentationRenderer> dilationRenderer) {
    try {
      std::string baseName = fs::path(filename).stem().string();

      // Export original
      renderToImage->removeAllRenderers();
      renderToImage->connect(originalRenderer);
      renderToImage->update();
      auto exporter = ImageFileExporter::create(outputPath + "/" + baseName +
                                                "_original.jpg");
      exporter->connect(renderToImage->getOutputData<Image>(0));
      exporter->update();

      // Export final result
      renderToImage->removeAllRenderers();
      renderToImage->connect(dilationRenderer);
      renderToImage->update();
      exporter = ImageFileExporter::create(outputPath + "/" + baseName +
                                           "_processed.jpg");
      exporter->connect(renderToImage->getOutputData<Image>(0));
      exporter->update();
    } catch (Exception &e) {
      std::cerr << "Error in export stage: " << e.what() << std::endl;
      throw;
    }
  }

public:
  SequentialImageProcessor(const std::string &outputDir = "../out-sequential")
      : outputPath(outputDir) {
    basePath = Config::getTestDataPath() +
               "Brain-Tumor-Progression/T1-Post-Combined-P001-P020/PGBM-017/"
               "16.000000-T1post-19554/";

    setupOutputDirectory();
    loadDICOMFiles();
  }

  void loadDICOMFiles() {
    try {
      std::vector<std::pair<std::string, int>> fileNumberPairs;

      for (const auto &entry : fs::directory_iterator(basePath)) {
        if (entry.path().extension() == ".dcm") {
          std::string filepath = entry.path().string();
          int fileNumber = extractFileNumber(entry.path().filename().string());
          fileNumberPairs.push_back({filepath, fileNumber});
        }
      }

      std::sort(
          fileNumberPairs.begin(), fileNumberPairs.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });

      dicomFiles.clear();
      for (const auto &pair : fileNumberPairs) {
        dicomFiles.push_back(pair.first);
      }

      std::cout << "Found " << dicomFiles.size()
                << " DICOM files in: " << basePath << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error loading DICOM files: " << e.what() << std::endl;
      throw;
    }
  }

  void processSingleImage(const std::string &filename) {
    try {
      std::cout << "Processing: " << fs::path(filename).filename() << std::endl;

      // Import Stage
      auto importer = DICOMFileImporter::create(filename);
      importer->setLoadSeries(false);
      importer->update();

      // Get image dimensions to adjust seed points accordingly
      auto importedImage = importer->getOutputData<Image>(0);
      if (!importedImage) {
        throw Exception("Failed to get imported image");
      }

      int width = importedImage->getWidth();
      int height = importedImage->getHeight();

      // Safety check for minimum dimensions
      if (width < 100 || height < 100) {
        throw Exception("Image dimensions too small: " + std::to_string(width) +
                        "x" + std::to_string(height));
      }

      // Preprocessing Stage
      auto normalize =
          IntensityNormalization::create(0.5f, 2.5f, 0.0f, 10000.0f);
      normalize->connect(importer);
      normalize->update();

      auto clipping = IntensityClipping::create(0.68f, 4000.0f);
      clipping->connect(normalize);
      clipping->update();

      auto medianfilter = VectorMedianFilter::create(7);
      medianfilter->connect(clipping);
      medianfilter->update();

      auto sharpen = ImageSharpening::create(2.0f, 0.5f, 9);
      sharpen->connect(medianfilter);
      sharpen->update();

      // Segmentation Stage
      // Calculate center and adjust seed points based on image dimensions
      int centerX = width / 2;
      int centerY = height / 2;

      // Create seed points relative to center
      std::vector<Vector3i> seedPoints;

      // Add central seed point
      seedPoints.push_back(Vector3i(centerX, centerY, 0));

      // Add seed points around center
      int offsetX = width / 8;
      int offsetY = height / 8;

      seedPoints.push_back(Vector3i(centerX + offsetX, centerY, 0));
      seedPoints.push_back(Vector3i(centerX - offsetX, centerY, 0));
      seedPoints.push_back(Vector3i(centerX, centerY + offsetY, 0));
      seedPoints.push_back(Vector3i(centerX, centerY - offsetY, 0));

      auto regionGrowing =
          SeededRegionGrowing::create(0.74f, 0.91f, seedPoints);
      regionGrowing->connect(sharpen);

      // Add additional seed points in a grid pattern
      for (int x = width / 4; x < width * 3 / 4; x += width / 10) {
        for (int y = height / 4; y < height * 3 / 4; y += height / 10) {
          regionGrowing->addSeedPoint(x, y);
        }
      }

      regionGrowing->update();

      // Post-processing Stage
      auto caster = ImageCaster::create(TYPE_UINT8);
      caster->connect(regionGrowing);
      caster->update();

      auto dilation = Dilation::create(3);
      dilation->connect(caster);
      dilation->update();

      // Export Stage
      LabelColors labelColors;
      labelColors[1] = Color::White();

      auto renderToImage = RenderToImage::create(Color::Black(), 512, 512);
      auto originalRenderer = ImageRenderer::create()->connect(importer);
      auto dilationRenderer =
          SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
              ->connect(dilation);

      exportProcessedImage(filename, renderToImage, originalRenderer,
                           dilationRenderer);

    } catch (Exception &e) {
      std::cerr << "Error processing file " << filename << ":\n"
                << "Detailed error: " << e.what() << std::endl;
      // Don't throw here - allow processing of other images to continue
    }
  }

  void processAllImages() {
    // Set reporting method for different types of messages
    Reporter::setGlobalReportMethod(Reporter::INFO,
                                    Reporter::NONE); // Disable INFO messages
    Reporter::setGlobalReportMethod(Reporter::WARNING,
                                    Reporter::COUT); // Keep warnings to console
    Reporter::setGlobalReportMethod(Reporter::ERROR,
                                    Reporter::COUT); // Keep errors to console

    std::cout << "\n=== Starting Sequential Processing ===\n" << std::endl;
    std::cout << "Found " << dicomFiles.size() << " images to process"
              << std::endl;

    int successCount = 0;
    for (size_t i = 0; i < dicomFiles.size(); ++i) {
      try {
        processSingleImage(dicomFiles[i]);
        successCount++;
      } catch (const std::exception &e) {
        std::cerr << "Failed to process image " << (i + 1)
                  << ". Moving to next image." << std::endl;
      }
    }

    std::cout << "\nProcessing completed. Successfully processed "
              << successCount << "/" << dicomFiles.size() << " images."
              << std::endl;
  }
};

int main() {
  try {
    // Set reporting method for different types of messages
    Reporter::setGlobalReportMethod(Reporter::INFO,
                                    Reporter::NONE); // Disable INFO messages
    Reporter::setGlobalReportMethod(Reporter::WARNING,
                                    Reporter::COUT); // Keep warnings to console
    Reporter::setGlobalReportMethod(Reporter::ERROR,
                                    Reporter::COUT); // Keep errors to console

    SequentialImageProcessor processor;
    processor.processAllImages();
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
