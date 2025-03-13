#include "FAST/includes.hpp"
#include <chrono>
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

  // Timing measurements
  std::chrono::duration<double> importTime{0};
  std::chrono::duration<double> preprocessTime{0};
  std::chrono::duration<double> segmentationTime{0};
  std::chrono::duration<double> postprocessTime{0};
  std::chrono::duration<double> exportTime{0};
  std::chrono::duration<double> totalTime{0};

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
  void verifyProcessingStep(std::shared_ptr<ProcessObject> process,
                            const std::string &stepName) {
    try {
      process->update();
      auto output = process->getOutputData<Image>(0);
      if (!output) {
        throw Exception("No output data produced at " + stepName);
      }
    } catch (Exception &e) {
      std::cerr << "Error at " << stepName << ": " << e.what() << std::endl;
      throw;
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
    basePath = "/home/rayyan/Capstone/Formatted-Dataset/T1-Post-Combined-P001-P020";

    setupOutputDirectory();
    loadDICOMFiles();
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

  void processSingleImage(const std::string &filename) {
    auto startTotal = std::chrono::high_resolution_clock::now();
    try {
        std::cout << "Processing: " << fs::path(filename).filename() << std::endl;
        // Import Stage
        auto startImport = std::chrono::high_resolution_clock::now();
        auto importer = DICOMFileImporter::create(filename);
        importer->setLoadSeries(false);
        verifyProcessingStep(importer, "Import Stage");
        importTime += std::chrono::high_resolution_clock::now() - startImport;

        // Preprocessing Stage
        auto startPreprocess = std::chrono::high_resolution_clock::now();
        auto normalize =
            IntensityNormalization::create(0.5f, 2.5f, 0.0f, 10000.0f);
        normalize->connect(importer);
        verifyProcessingStep(normalize, "Normalization");
        auto clipping = IntensityClipping::create(0.68f, 4000.0f);
        clipping->connect(normalize);
        verifyProcessingStep(clipping, "Clipping");
        auto medianfilter = VectorMedianFilter::create(5);
        medianfilter->connect(clipping);
        verifyProcessingStep(medianfilter, "Median Filter");
        auto sharpen = ImageSharpening::create(2.0f, 0.5f, 9);
        sharpen->connect(medianfilter);
        verifyProcessingStep(sharpen, "Sharpening");
        preprocessTime +=
            std::chrono::high_resolution_clock::now() - startPreprocess;

        // Get image dimensions
        auto originalImage = importer->getOutputData<Image>(0);
        int width = originalImage->getWidth();
        int height = originalImage->getHeight();

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

        // Add additional seed points in a grid pattern
        for (int x = std::max(0, width / 4); x < width * 3 / 4; x += width / 8) {
            for (int y = std::max(0, height / 4); y < height * 3 / 4; y += height / 8) {
                regionGrowing->addSeedPoint(x, y);
            }
        }

        verifyProcessingStep(regionGrowing, "Region Growing");
        segmentationTime +=
            std::chrono::high_resolution_clock::now() - startSegmentation;

        // Post-processing Stage
        auto startPostprocess = std::chrono::high_resolution_clock::now();
        auto caster = ImageCaster::create(TYPE_UINT8);
        caster->connect(regionGrowing);
        verifyProcessingStep(caster, "Type Casting");
        auto dilation = Dilation::create(3);
        dilation->connect(caster);
        verifyProcessingStep(dilation, "Dilation");
        postprocessTime +=
            std::chrono::high_resolution_clock::now() - startPostprocess;

        // Export Stage
        auto startExport = std::chrono::high_resolution_clock::now();
        LabelColors labelColors;
        labelColors[1] = Color::White();
        auto renderToImage = RenderToImage::create(Color::Black(), 512, 512);
        auto originalRenderer = ImageRenderer::create()->connect(importer);
        auto dilationRenderer =
            SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
                ->connect(dilation);
        exportProcessedImage(filename, renderToImage, originalRenderer,
                             dilationRenderer);
        exportTime += std::chrono::high_resolution_clock::now() - startExport;

        totalTime += std::chrono::high_resolution_clock::now() - startTotal;
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

    printTimingResults();
  }

  void printTimingResults() const {
    std::cout << "\n=== Processing Time Results ===\n" << std::endl;
    std::cout << "Import Time: " << importTime.count() << " seconds"
              << std::endl;
    std::cout << "Preprocessing Time: " << preprocessTime.count() << " seconds"
              << std::endl;
    std::cout << "Segmentation Time: " << segmentationTime.count() << " seconds"
              << std::endl;
    std::cout << "Post-processing Time: " << postprocessTime.count()
              << " seconds" << std::endl;
    std::cout << "Export Time: " << exportTime.count() << " seconds"
              << std::endl;
    double calculatedTotalTime = importTime.count() + preprocessTime.count() +
                                 segmentationTime.count() +
                                 postprocessTime.count() + exportTime.count();
    std::cout << "Calculated Total Time: " << calculatedTotalTime << " seconds" 
              << std::endl;
    std::cout << "Total Time: " << totalTime.count() << " seconds"
              << std::endl;
    std::cout << "Average Time per Image: "
              << totalTime.count() / dicomFiles.size() << " seconds"
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
