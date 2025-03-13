#include "FAST/includes.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

using namespace fast;
namespace fs = std::filesystem;
using json = nlohmann::json;

class SequentialImageProcessor {
private:
  std::string baseDataPath;
  std::string outputPath;
  json resultsJson;

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

  void setupOutputDirectory(const std::string &patientDir) {
    try {
      std::string patientOutputPath = outputPath + "/" + patientDir;
      if (system(("mkdir -p " + patientOutputPath).c_str()) != 0) {
        throw std::runtime_error("Failed to create output directory: " +
                                 patientOutputPath);
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Error setting up output directory: " +
                               std::string(e.what()));
    }
  }

  void clearOutputDirectory() {
    try {
      if (system(("mkdir -p " + outputPath + " && cd " + outputPath +
                  " && rm -rf *")
                     .c_str()) != 0) {
        throw std::runtime_error("Failed to create clean output directory: " +
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
                       std::shared_ptr<SegmentationRenderer> dilationRenderer,
                       const std::string &patientDir) {
    try {
      std::string baseName = fs::path(filename).stem().string();
      std::string patientOutputPath = outputPath + "/" + patientDir;

      // Export original
      renderToImage->removeAllRenderers();
      renderToImage->connect(originalRenderer);
      renderToImage->update();
      auto exporter = ImageFileExporter::create(patientOutputPath + "/" +
                                                baseName + "_original.jpg");
      exporter->connect(renderToImage->getOutputData<Image>(0));
      exporter->update();

      // Export final result
      renderToImage->removeAllRenderers();
      renderToImage->connect(dilationRenderer);
      renderToImage->update();
      exporter = ImageFileExporter::create(patientOutputPath + "/" + baseName +
                                           "_processed.jpg");
      exporter->connect(renderToImage->getOutputData<Image>(0));
      exporter->update();
    } catch (Exception &e) {
      std::cerr << "Error in export stage: " << e.what() << std::endl;
      throw;
    }
  }

  std::vector<std::string>
  loadDICOMFilesForPatient(const std::string &patientPath) {
    std::vector<std::string> files;
    std::vector<std::pair<std::string, int>> fileNumberPairs;

    try {
      // Find the T1post directory
      for (const auto &sessionDir : fs::directory_iterator(patientPath)) {
        if (!fs::is_directory(sessionDir))
          continue;

        std::string dirName = sessionDir.path().filename().string();
        if (dirName.find("T1post") != std::string::npos) {
          // Found T1post directory, now get all DCM files
          for (const auto &entry : fs::directory_iterator(sessionDir.path())) {
            if (entry.path().extension() == ".dcm") {
              std::string filepath = entry.path().string();
              int fileNumber =
                  extractFileNumber(entry.path().filename().string());
              fileNumberPairs.push_back({filepath, fileNumber});
            }
          }
          break;
        }
      }

      std::sort(
          fileNumberPairs.begin(), fileNumberPairs.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });

      for (const auto &pair : fileNumberPairs) {
        files.push_back(pair.first);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error loading DICOM files: " << e.what() << std::endl;
    }

    return files;
  }

public:
  SequentialImageProcessor(const std::string &outputDir = "../out-sequential")
      : outputPath(outputDir) {
    baseDataPath = Config::getTestDataPath() +
                   "Brain-Tumor-Progression/T1-Post-Combined-P001-P020/";

    // Create main output directory
    if (system(("mkdir -p " + outputPath).c_str()) != 0) {
      throw std::runtime_error("Failed to create main output directory: " +
                               outputPath);
    }

    // Initialize JSON structure
    resultsJson["processor"] = "sequential";
    resultsJson["patients"] = json::array();
  }

  void processSingleImage(const std::string &filename,
                          const std::string &patientDir) {
    auto startTotal = std::chrono::high_resolution_clock::now();

    try {
      // Import Stage
      auto startImport = std::chrono::high_resolution_clock::now();
      auto importer = DICOMFileImporter::create(filename);
      importer->setLoadSeries(false);
      verifyProcessingStep(importer, "Import Stage");
      importTime += std::chrono::high_resolution_clock::now() - startImport;

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

      auto renderToImage = RenderToImage::create(Color::Black(), width, height);
      auto originalRenderer = ImageRenderer::create()->connect(importer);
      auto dilationRenderer =
          SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
              ->connect(dilation);

      exportProcessedImage(filename, renderToImage, originalRenderer,
                           dilationRenderer, patientDir);

      exportTime += std::chrono::high_resolution_clock::now() - startExport;

    } catch (Exception &e) {
      std::cerr << "Error processing file " << filename << ":\n"
                << "Detailed error: " << e.what() << std::endl;
      // Don't throw here - allow processing of other images to continue
    }

    totalTime += std::chrono::high_resolution_clock::now() - startTotal;
  }

  void processAllPatients() {
    // Set reporting method for different types of messages
    Reporter::setGlobalReportMethod(Reporter::INFO,
                                    Reporter::NONE); // Disable INFO messages
    Reporter::setGlobalReportMethod(Reporter::WARNING,
                                    Reporter::COUT); // Keep warnings to console
    Reporter::setGlobalReportMethod(Reporter::ERROR,
                                    Reporter::COUT); // Keep errors to console
    // Clear out-sequential directory
    clearOutputDirectory();

    std::cout << "\n=== Starting Sequential Processing ===\n" << std::endl;

    // Process each patient directory
    for (const auto &patientEntry : fs::directory_iterator(baseDataPath)) {
      if (!fs::is_directory(patientEntry))
        continue;

      std::string patientDir = patientEntry.path().filename().string();

      // Skip if not a PGBM-XXX directory
      if (patientDir.substr(0, 5) != "PGBM-")
        continue;

      // Reset timing measurements for this patient
      importTime = std::chrono::duration<double>{0};
      preprocessTime = std::chrono::duration<double>{0};
      segmentationTime = std::chrono::duration<double>{0};
      postprocessTime = std::chrono::duration<double>{0};
      exportTime = std::chrono::duration<double>{0};
      totalTime = std::chrono::duration<double>{0};

      // Setup output directory for this patient
      setupOutputDirectory(patientDir);

      // Load DICOM files for this patient
      std::vector<std::string> dicomFiles =
          loadDICOMFilesForPatient(patientEntry.path().string());

      std::cout << "Processing " << patientDir << ": Found "
                << dicomFiles.size() << " images" << std::endl;

      int successCount = 0;
      for (size_t i = 0; i < dicomFiles.size(); ++i) {
        try {
          processSingleImage(dicomFiles[i], patientDir);
          successCount++;
        } catch (const std::exception &e) {
          std::cerr << "Failed to process image " << (i + 1) << " for patient "
                    << patientDir << ". Moving to next image." << std::endl;
        }
      }

      // Collect results for this patient
      json patientResults;
      patientResults["patient_id"] = patientDir;
      patientResults["total_images"] = dicomFiles.size();
      patientResults["successful_images"] = successCount;
      patientResults["timing"] = {
          {"import_time", importTime.count()},
          {"preprocessing_time", preprocessTime.count()},
          {"segmentation_time", segmentationTime.count()},
          {"postprocessing_time", postprocessTime.count()},
          {"export_time", exportTime.count()},
          {"total_time", totalTime.count()},
          {"average_time_per_image",
           dicomFiles.empty() ? 0 : totalTime.count() / dicomFiles.size()}};

      // Add to JSON results
      resultsJson["patients"].push_back(patientResults);

      // Output results for this patient
      std::cout << "\n=== Results for " << patientDir << " ===\n";
      std::cout << "Successfully processed " << successCount << "/"
                << dicomFiles.size() << " images" << std::endl;
      printTimingResults(dicomFiles.size());
      std::cout << std::endl;
    }

    // Write results to JSON file
    std::string projectRootDir = "../";
    std::ofstream jsonFile(projectRootDir + "sequential_results.json");
    jsonFile << std::setw(4) << resultsJson << std::endl;
    jsonFile.close();

    std::cout
        << "\nAll patients processed. Results saved to sequential_results.json"
        << std::endl;
  }

  void printTimingResults(size_t imageCount) const {
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
              << (imageCount > 0 ? totalTime.count() / imageCount : 0)
              << " seconds" << std::endl;
  }
};

int main() {
  try {
    // Create sequential processor
    SequentialImageProcessor processor;

    // Process all patients
    processor.processAllPatients();

    std::cout << "\nSequential processing completed successfully." << std::endl;

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error in main: " << e.what() << std::endl;
    return -1;
  }
}
