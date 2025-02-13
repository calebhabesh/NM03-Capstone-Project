#include "FAST/Visualization/RenderToImage/RenderToImage.hpp"
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
      // Remove and recreate directory in one command, similar to test code
      if (system(("rm -rf " + outputPath + " && mkdir -p " + outputPath)
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
      std::cout << "Successfully completed: " << stepName << std::endl;
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
    basePath = Config::getTestDataPath() +
               "Brain-Tumor-Progression/PGBM-017/09-17-1997-RA FH MR RCBV "
               "OP-85753/16.000000-T1post-19554/";

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
    auto startTotal = std::chrono::high_resolution_clock::now();

    try {
      std::cout << "\nProcessing: " << fs::path(filename).filename()
                << std::endl;

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

      // Segmentation Stage
      auto startSegmentation = std::chrono::high_resolution_clock::now();

      auto regionGrowing =
          SeededRegionGrowing::create(0.74f, 0.91f,
                                      std::vector<Vector3i>{{300, 256, 0},
                                                            {320, 256, 0},
                                                            {340, 256, 0},
                                                            {300, 236, 0},
                                                            {300, 276, 0},
                                                            {212, 256, 0},
                                                            {192, 256, 0},
                                                            {172, 256, 0},
                                                            {212, 236, 0},
                                                            {212, 276, 0}});
      regionGrowing->connect(sharpen);
      // Add additional seed points in a grid pattern

      for (int x = 150; x < 362; x += 30) {
        for (int y = 150; y < 362; y += 30) {
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

    } catch (Exception &e) {
      std::cerr << "Error processing file " << filename << ":\n"
                << "Detailed error: " << e.what() << std::endl;
      // Don't throw here - allow processing of other images to continue
    }

    totalTime += std::chrono::high_resolution_clock::now() - startTotal;
  }

  void processAllImages() {
    std::cout << "Starting sequential processing of " << dicomFiles.size()
              << " images..." << std::endl;

    int successCount = 0;
    for (size_t i = 0; i < dicomFiles.size(); ++i) {
      try {
        std::cout << "\nProcessing image " << (i + 1) << "/"
                  << dicomFiles.size() << ": "
                  << fs::path(dicomFiles[i]).filename().string() << std::endl;
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
    std::cout << "\nProcessing Time Results:" << std::endl;
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
    std::cout << "Total Time: " << totalTime.count() << " seconds" << std::endl;
    std::cout << "Average Time per Image: "
              << totalTime.count() / dicomFiles.size() << " seconds"
              << std::endl;
  }
};

int main() {
  try {
    SequentialImageProcessor processor;
    processor.processAllImages();
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
