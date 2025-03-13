#include "FAST/includes.hpp"
#include <atomic>
#include <chrono>
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

struct TimingData {
  double importTime = 0;
  double preprocessTime = 0;
  double segmentationTime = 0;
  double postprocessTime = 0;
  double totalTime = 0;
};

class OptimizedParallelProcessor {
private:
  std::vector<std::string> dicomFiles;
  std::string basePath;
  std::string outputPath;
  std::mutex outputMutex;
  std::vector<TimingData> threadTimings;
  double exportTime = 0;

  std::shared_ptr<RenderToImage> renderToImage;
  std::atomic<size_t> completedImages{0};
  const size_t PROGRESS_REPORT_INTERVAL = 5;
  static const size_t DEFAULT_BATCH_SIZE = 5;

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

  void verifyProcessingStep(std::shared_ptr<ProcessObject> process,
                            const std::string &stepName,
                            const std::string &filename) {
    try {
      process->update();
      auto output = process->getOutputData<Image>(0);
      if (!output) {
        throw Exception("No output data produced at " + stepName);
      }
      // Success logging removed to reduce overhead
    } catch (Exception &e) {
      std::lock_guard<std::mutex> lock(outputMutex);
      std::cerr << "Error at " << stepName << " for " << filename << ": "
                << e.what() << std::endl;
      throw;
    }
  }

  ProcessedImageData processSingleImage(const std::string &filename,
                                      TimingData &timing) {
    ProcessedImageData result;
    result.filename = filename;

#pragma omp critical
    {
        std::cout << "Processing: \"" << fs::path(filename).filename().string()
                  << "\"" << std::endl;
    }

    auto startTotal = std::chrono::high_resolution_clock::now();
    try {
        // Import Stage
        auto startImport = std::chrono::high_resolution_clock::now();
        auto importer = DICOMFileImporter::create(filename);
        importer->setLoadSeries(false);
        verifyProcessingStep(importer, "Import", filename);
        timing.importTime +=
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startImport)
                .count();
        result.originalImage = importer->getOutputData<Image>(0);

        // Get image dimensions
        int width = result.originalImage->getWidth();
        int height = result.originalImage->getHeight();

        // Preprocessing Stage
        auto startPreprocess = std::chrono::high_resolution_clock::now();
        auto normalize =
            IntensityNormalization::create(0.5f, 2.5f, 0.0f, 10000.0f);
        normalize->connect(importer);
        auto clipping = IntensityClipping::create(0.68f, 4000.0f);
        clipping->connect(normalize);
        auto medianfilter = VectorMedianFilter::create(5);
        medianfilter->connect(clipping);
        auto sharpen = ImageSharpening::create(2.0f, 0.5f, 9);
        sharpen->connect(medianfilter);
        verifyProcessingStep(sharpen, "Preprocessing", filename);
        timing.preprocessTime +=
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startPreprocess)
                .count();

        // Segmentation Stage
        auto startSegmentation = std::chrono::high_resolution_clock::now();

        // Dynamically generate seed points based on image dimensions
        std::vector<Vector3i> seedPoints;
        int centerX = width / 2;
        int centerY = height / 2;
        int radius = std::min(width, height) / 4; // Adjust radius dynamically

        for (int x = centerX - radius; x <= centerX + radius; x += radius / 2) {
            for (int y = centerY - radius; y <= centerY + radius; y += radius / 2) {
                // Clamp seed points to image boundaries
                int clampedX = std::max(0, std::min(x, width - 1));
                int clampedY = std::max(0, std::min(y, height - 1));
                seedPoints.emplace_back(clampedX, clampedY, 0);
            }
        }

        auto regionGrowing = SeededRegionGrowing::create(0.74f, 0.91f, seedPoints);
        regionGrowing->connect(sharpen);
        verifyProcessingStep(regionGrowing, "Segmentation", filename);
        timing.segmentationTime +=
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startSegmentation)
                .count();

        // Post-processing Stage
        auto startPostprocess = std::chrono::high_resolution_clock::now();
        auto caster = ImageCaster::create(TYPE_UINT8);
        caster->connect(regionGrowing);
        auto dilation = Dilation::create(3);
        dilation->connect(caster);
        verifyProcessingStep(dilation, "Post-processing", filename);
        result.processedImage = dilation->getOutputData<Image>(0);
        timing.postprocessTime +=
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startPostprocess)
                .count();
        timing.totalTime +=
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startTotal)
                .count();
    } catch (Exception &e) {
        std::lock_guard<std::mutex> lock(outputMutex);
        std::cerr << "Error processing file " << filename << ": "
                  << "Detailed error: " << e.what() << std::endl;
    }
    return result;
}

  void exportBatch(const std::vector<ProcessedImageData> &batch) {
    auto startExport = std::chrono::high_resolution_clock::now();

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

    exportTime += std::chrono::duration<double>(
                      std::chrono::high_resolution_clock::now() - startExport)
                      .count();
  }

public:
  OptimizedParallelProcessor(const std::string &outputDir = "../out-parallel")
      : outputPath(outputDir) {
    basePath = "/home/rayyan/Capstone/Formatted-Dataset/T1-Post-Combined-P001-P020";

    setupOutputDirectory();
    loadDICOMFiles();
    threadTimings.resize(omp_get_max_threads());
    renderToImage = RenderToImage::create(Color::Black(), 512, 512);
  }

  void loadDICOMFiles() {
    try {
      std::vector<std::pair<std::string, int>> fileNumberPairs;
      // Use recursive_directory_iterator to traverse subdirectories 
      for (const auto &entry : fs::recursive_directory_iterator(basePath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".dcm") {
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

#pragma omp parallel for schedule(static, 1) reduction(+ : successCount)
      for (size_t i = 0; i < currentBatchSize; ++i) {
        size_t fileIndex = batchStart + i;
        batchResults[i] = processSingleImage(
            dicomFiles[fileIndex], threadTimings[omp_get_thread_num()]);
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

    printTimingResults();
  }

  void printTimingResults() const {
    TimingData totalTiming;
    for (const auto &timing : threadTimings) {
      totalTiming.importTime += timing.importTime;
      totalTiming.preprocessTime += timing.preprocessTime;
      totalTiming.segmentationTime += timing.segmentationTime;
      totalTiming.postprocessTime += timing.postprocessTime;
      totalTiming.totalTime += timing.totalTime;
    }

    std::cout << "\n=== Processing Time Results ===\n" << std::endl;
    std::cout << "Import Time: " << totalTiming.importTime << " seconds"
              << std::endl;
    std::cout << "Preprocessing Time: " << totalTiming.preprocessTime
              << " seconds" << std::endl;
    std::cout << "Segmentation Time: " << totalTiming.segmentationTime
              << " seconds" << std::endl;
    std::cout << "Post-processing Time: " << totalTiming.postprocessTime
              << " seconds" << std::endl;
    std::cout << "Export Time: " << exportTime << " seconds" << std::endl;
    std::cout << "Total Time: " << totalTiming.totalTime << " seconds"
              << std::endl;
    std::cout << "Average Time per Image: "
              << totalTiming.totalTime / dicomFiles.size() << " seconds"
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

    OptimizedParallelProcessor processor;
    processor.processAllImages();

  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
