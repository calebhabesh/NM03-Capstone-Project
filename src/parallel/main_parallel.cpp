#include "FAST/includes.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <vector>

using namespace fast;
namespace fs = std::filesystem;
using json = nlohmann::json;

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
  double exportTime = 0;
  double totalTime = 0;
  int totalImages = 0;
  int successfulImages = 0;
};

class OptimizedParallelProcessor {
private:
  std::string baseDataPath;
  std::string outputPath;
  std::mutex outputMutex;
  std::map<std::string, TimingData> patientTimings; // Timing per patient
  json resultsJson;

  std::shared_ptr<RenderToImage> renderToImage;
  // Adjust the batch size here
  // Tuning this has an effect on the processing time, this is analyzed in
  // Python Ex: Batch Size = 4, Num Scans = 23,
  // Batches = 4 + 4 + 4 + 4 + 4 + 3 = 23
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
      if (system(("mkdir -p " + outputPath).c_str()) != 0) {
        throw std::runtime_error("Failed to create main output directory: " +
                                 outputPath);
      }

      // Clear output directory
      if (system(("cd " + outputPath + " && rm -rf *").c_str()) != 0) {
        throw std::runtime_error("Failed to clean output directory: " +
                                 outputPath);
      }

      std::cout << "Created and cleaned output directory: " + outputPath
                << std::endl;
    } catch (const std::exception &e) {
      throw std::runtime_error("Error setting up output directory: " +
                               std::string(e.what()));
    }
  }

  void setupPatientOutputDirectory(const std::string &patientDir) {
    try {
      std::string patientOutputPath = outputPath + "/" + patientDir;
      if (system(("mkdir -p " + patientOutputPath).c_str()) != 0) {
        throw std::runtime_error("Failed to create patient output directory: " +
                                 patientOutputPath);
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Error setting up patient output directory: " +
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

      timing.successfulImages++;

    } catch (Exception &e) {
      std::lock_guard<std::mutex> lock(outputMutex);
      std::cerr << "Error processing file " << filename << ":\n"
                << "Detailed error: " << e.what() << std::endl;
    }

    return result;
  }

  void exportBatch(const std::vector<ProcessedImageData> &batch,
                   const std::string &patientDir, TimingData &timing) {
    auto startExport = std::chrono::high_resolution_clock::now();

    try {
      LabelColors labelColors;
      labelColors[1] = Color::White();

      for (const auto &imageData : batch) {
        if (!imageData.originalImage || !imageData.processedImage) {
          continue;
        }

        std::string baseName = fs::path(imageData.filename).stem().string();
        std::string patientOutputPath = outputPath + "/" + patientDir;

        // Export original
        {
          renderToImage->removeAllRenderers();
          auto originalRenderer = ImageRenderer::create();
          originalRenderer->addInputData(imageData.originalImage);
          renderToImage->connect(originalRenderer);
          renderToImage->update();

          auto exporter = ImageFileExporter::create(patientOutputPath + "/" +
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
              patientOutputPath + "/" + baseName + "_processed.jpg");
          exporter->connect(renderToImage->getOutputData<Image>(0));
          exporter->update();
        }
      }
    } catch (Exception &e) {
      std::cerr << "Error in export stage: " << e.what() << std::endl;
    }

    timing.exportTime +=
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - startExport)
            .count();
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
  OptimizedParallelProcessor(const std::string &outputDir = "../out-parallel")
      : outputPath(outputDir) {
    baseDataPath = Config::getTestDataPath() +
                   "Brain-Tumor-Progression/T1-Post-Combined-P001-P020/";

    setupOutputDirectory();
    renderToImage = RenderToImage::create(Color::Black(), 512, 512);

    // Initialize JSON structure
    resultsJson["processor"] = "parallel";
    resultsJson["patients"] = json::array();
  }

  void processPatient(const std::string &patientDir,
                      size_t batchSize = DEFAULT_BATCH_SIZE) {
    std::string patientPath = baseDataPath + "/" + patientDir;

    // Setup output directory for this patient
    setupPatientOutputDirectory(patientDir);

    // Reset timing for this patient
    patientTimings[patientDir] = TimingData();

    // Load DICOM files for this patient
    std::vector<std::string> dicomFiles = loadDICOMFilesForPatient(patientPath);
    patientTimings[patientDir].totalImages = dicomFiles.size();

    std::cout << "Processing " << patientDir << ": Found " << dicomFiles.size()
              << " images" << std::endl;

    if (dicomFiles.empty()) {
      std::cout << "No DICOM files found for patient " << patientDir
                << std::endl;
      return;
    }

    // Process images in batches
    for (size_t batchStart = 0; batchStart < dicomFiles.size();
         batchStart += batchSize) {
      size_t currentBatchSize =
          std::min(batchSize, dicomFiles.size() - batchStart);
      std::vector<ProcessedImageData> batchResults(currentBatchSize);

#pragma omp parallel for schedule(static, 1)
      for (size_t i = 0; i < currentBatchSize; ++i) {
        size_t fileIndex = batchStart + i;
        // Removed the per-file output inside the critical section
        batchResults[i] = processSingleImage(dicomFiles[fileIndex],
                                             patientTimings[patientDir]);
      }

      // Export batch results
      exportBatch(batchResults, patientDir, patientTimings[patientDir]);
    }

    // Collect results for this patient
    json patientResults;
    patientResults["patient_id"] = patientDir;
    patientResults["total_images"] = patientTimings[patientDir].totalImages;
    patientResults["successful_images"] =
        patientTimings[patientDir].successfulImages;
    patientResults["timing"] = {
        {"import_time", patientTimings[patientDir].importTime},
        {"preprocessing_time", patientTimings[patientDir].preprocessTime},
        {"segmentation_time", patientTimings[patientDir].segmentationTime},
        {"postprocessing_time", patientTimings[patientDir].postprocessTime},
        {"export_time", patientTimings[patientDir].exportTime},
        {"total_time", patientTimings[patientDir].totalTime},
        {"average_time_per_image",
         dicomFiles.empty()
             ? 0
             : patientTimings[patientDir].totalTime / dicomFiles.size()}};

    // Add to JSON results
    resultsJson["patients"].push_back(patientResults);

    // Output results for this patient
    std::cout << "\n=== Results for " << patientDir << " ===\n";
    std::cout << "Successfully processed "
              << patientTimings[patientDir].successfulImages << "/"
              << dicomFiles.size() << " images" << std::endl;
    printPatientTimingResults(patientDir);
    std::cout << std::endl;
  }

  void processAllPatients(size_t batchSize = DEFAULT_BATCH_SIZE) {
    std::cout << "\n=== Starting Parallel Processing on all patients ===\n"
              << std::endl;
    std::cout << "Using " << omp_get_max_threads() << " threads\n" << std::endl;

    // Process each patient directory
    for (const auto &patientEntry : fs::directory_iterator(baseDataPath)) {
      if (!fs::is_directory(patientEntry))
        continue;

      std::string patientDir = patientEntry.path().filename().string();

      // Skip if not a PGBM-XXX directory
      if (patientDir.substr(0, 5) != "PGBM-")
        continue;

      processPatient(patientDir, batchSize);
    }

    // Write results to JSON file
    std::string projectRootDir = "../";
    std::ofstream jsonFile(projectRootDir + "parallel_results.json");
    jsonFile << std::setw(4) << resultsJson << std::endl;
    jsonFile.close();

    std::cout
        << "\nAll patients processed. Results saved to parallel_results.json"
        << std::endl;
  }

  void printPatientTimingResults(const std::string &patientDir) const {
    const auto &timing = patientTimings.at(patientDir);
    std::cout << "Import Time: " << timing.importTime << " seconds"
              << std::endl;
    std::cout << "Preprocessing Time: " << timing.preprocessTime << " seconds"
              << std::endl;
    std::cout << "Segmentation Time: " << timing.segmentationTime << " seconds"
              << std::endl;
    std::cout << "Post-processing Time: " << timing.postprocessTime
              << " seconds" << std::endl;
    std::cout << "Export Time: " << timing.exportTime << " seconds"
              << std::endl;
    std::cout << "Total Time: " << timing.totalTime << " seconds" << std::endl;
    std::cout << "Average Time per Image: "
              << (timing.totalImages > 0 ? timing.totalTime / timing.totalImages
                                         : 0)
              << " seconds" << std::endl;
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
    processor.processAllPatients();

  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
