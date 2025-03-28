#include "FAST/includes.hpp"
#include <atomic>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <vector>

using namespace fast;
namespace fs = std::filesystem;

// Structure to hold processed image data
struct ProcessedImageData {
  std::string filename;
  std::shared_ptr<Image> originalImage;
  std::shared_ptr<Image> processedImage;
};

class OptimizedParallelProcessor {
private:
  std::vector<std::string> dicomFiles;
  std::string basePath;
  std::string outputPath;
  std::mutex outputMutex;
  std::shared_ptr<RenderToImage> renderToImage;
  std::atomic<size_t> completedImages{0};
  const size_t PROGRESS_REPORT_INTERVAL = 5;
  static const size_t DEFAULT_BATCH_SIZE = 16;

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
      std::cout << "Created output directory: " + outputPath << std::endl;
    } catch (const std::exception &e) {
      throw std::runtime_error("Error setting up output directory: " +
                               std::string(e.what()));
    }
  }

  ProcessedImageData processSingleImage(const std::string &filename) {
    ProcessedImageData result;
    result.filename = filename;

#pragma omp critical
    {
      std::cout << "Processing: \"" << fs::path(filename).filename().string()
                << "\"" << std::endl;
    }

    try {
      // Import Stage
      auto importer = DICOMFileImporter::create(filename);
      importer->setLoadSeries(false);
      importer->update();

      result.originalImage = importer->getOutputData<Image>(0);

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
#pragma omp parallel for collapse(2) if (omp_get_level() < 2)
      for (int x = width / 4; x < width * 3 / 4; x += width / 10) {
        for (int y = height / 4; y < height * 3 / 4; y += height / 10) {
#pragma omp critical
          {
            regionGrowing->addSeedPoint(x, y);
          }
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

      result.processedImage = dilation->getOutputData<Image>(0);

    } catch (Exception &e) {
      std::lock_guard<std::mutex> lock(outputMutex);
      std::cerr << "Error processing file " << filename << ":\n"
                << "Detailed error: " << e.what() << std::endl;
    }

    return result;
  }

  void exportBatch(const std::vector<ProcessedImageData> &batch) {
    try {
      LabelColors labelColors;
      labelColors[1] = Color::White();

      for (const auto &imageData : batch) {
        if (!imageData.originalImage || !imageData.processedImage) {
          continue;
        }

        std::string baseName = fs::path(imageData.filename).stem().string();

        // Export original
        {
          renderToImage->removeAllRenderers();
          auto originalRenderer = ImageRenderer::create();
          originalRenderer->addInputData(imageData.originalImage);
          renderToImage->connect(originalRenderer);
          renderToImage->update();

          auto exporter = ImageFileExporter::create(outputPath + "/" +
                                                    baseName + "_original.jpg");
          exporter->connect(renderToImage->getOutputData<Image>(0));
          exporter->update();
        }

        // Export processed
        {
          renderToImage->removeAllRenderers();
          auto processedRenderer =
              SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2);
          processedRenderer->addInputData(imageData.processedImage);
          renderToImage->connect(processedRenderer);
          renderToImage->update();

          auto exporter = ImageFileExporter::create(
              outputPath + "/" + baseName + "_processed.jpg");
          exporter->connect(renderToImage->getOutputData<Image>(0));
          exporter->update();
        }
      }
    } catch (Exception &e) {
      std::cerr << "Error in export stage: " << e.what() << std::endl;
    }
  }

public:
  OptimizedParallelProcessor(const std::string &outputDir = "../out-parallel")
      : outputPath(outputDir) {
    basePath = Config::getTestDataPath() +
               "Brain-Tumor-Progression/T1-Post-Combined-P001-P020/PGBM-017/"
               "16.000000-T1post-19554/";

    setupOutputDirectory();
    loadDICOMFiles();
    renderToImage = RenderToImage::create(Color::Black(), 512, 512);
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

  void processAllImages(size_t batchSize = DEFAULT_BATCH_SIZE) {
    std::cout << "\n=== Starting Parallel Processing ===\n" << std::endl;
    std::cout << "Found " << dicomFiles.size() << " images to process"
              << std::endl;
    std::cout << "Using " << omp_get_max_threads() << " threads\n" << std::endl;

    int successCount = 0;

    // Process images in batches
    for (size_t batchStart = 0; batchStart < dicomFiles.size();
         batchStart += batchSize) {
      size_t currentBatchSize =
          std::min(batchSize, dicomFiles.size() - batchStart);
      std::vector<ProcessedImageData> batchResults(currentBatchSize);

#pragma omp parallel for schedule(static) reduction(+ : successCount)
      for (size_t i = 0; i < currentBatchSize; ++i) {
        size_t fileIndex = batchStart + i;
        batchResults[i] = processSingleImage(dicomFiles[fileIndex]);
        if (batchResults[i].originalImage && batchResults[i].processedImage) {
          successCount++;
        }
      }

      // Export batch results
      exportBatch(batchResults);
    }

    std::cout << "\nProcessing completed. Successfully processed "
              << successCount << "/" << dicomFiles.size() << " images."
              << std::endl;
  }
};

int main(int argc, char *argv[]) {
  try {
    QApplication app(argc, argv);

    // Set reporting method for different types of messages
    Reporter::setGlobalReportMethod(Reporter::INFO,
                                    Reporter::NONE); // Disable INFO messages
    Reporter::setGlobalReportMethod(Reporter::WARNING,
                                    Reporter::COUT); // Keep warnings to console
    Reporter::setGlobalReportMethod(Reporter::ERROR,
                                    Reporter::COUT); // Keep errors to console

    omp_set_num_threads(16);

    omp_set_nested(1);
    omp_set_max_active_levels(2);

    OptimizedParallelProcessor processor;
    processor.processAllImages();

  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
