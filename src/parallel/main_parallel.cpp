#include "FAST/FAST_directives.hpp"
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
  std::string baseDataPath;
  std::string patientPath;
  std::string outputBasePath;
  std::string currentOutputPath;
  std::mutex outputMutex;
  std::shared_ptr<RenderToImage> renderToImage;
  std::atomic<size_t> completedImages{0};

  // Corresponds to the batches that are divided into worker threads
  // Patient datasets range between 21-25 .dcm files, so just set to largest
  // possible num of dcm files in a patient directory
  static const size_t DEFAULT_BATCH_SIZE = 25;

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

  void setupOutputDirectory(const std::string &patientID) {
    try {
      currentOutputPath = outputBasePath + "/" + patientID;
      if (system(("mkdir -p " + currentOutputPath + " && cd " +
                  currentOutputPath + " && rm -rf *")
                     .c_str()) != 0) {
        throw std::runtime_error("Failed to setup output directory: " +
                                 currentOutputPath);
      }
      std::cout << "Created output directory: " + currentOutputPath
                << std::endl;
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
      for (int x = width / 4; x < width * 3 / 4; x += width / 10) {
        for (int y = height / 4; y < height * 3 / 4; y += height / 10) {
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

          auto exporter = ImageFileExporter::create(currentOutputPath + "/" +
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
              currentOutputPath + "/" + baseName + "_processed.jpg");
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
      : outputBasePath(outputDir) {
    baseDataPath = Config::getTestDataPath() +
                   "Brain-Tumor-Progression/T1-Post-Combined-P001-P020/";

    // Create main output directory
    if (system(("mkdir -p " + outputBasePath).c_str()) != 0) {
      throw std::runtime_error("Failed to create base output directory: " +
                               outputBasePath);
    }

    renderToImage = RenderToImage::create(Color::Black(), 512, 512);
  }

  std::vector<std::string> findAllPatientDirectories() {
    std::vector<std::string> patientDirs;

    try {
      for (const auto &entry : fs::directory_iterator(baseDataPath)) {
        if (entry.is_directory()) {
          std::string dirName = entry.path().filename().string();
          // Check if it's a patient directory (starts with "PGBM-")
          if (dirName.find("PGBM-") == 0) {
            patientDirs.push_back(dirName);
          }
        }
      }

      // Sort patient directories
      std::sort(patientDirs.begin(), patientDirs.end());

      std::cout << "Found " << patientDirs.size() << " patient directories."
                << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error finding patient directories: " << e.what()
                << std::endl;
      throw;
    }

    return patientDirs;
  }

  void loadDICOMFilesForPatient(const std::string &patientID) {
    try {
      // First find all subdirectories in the patient folder
      patientPath = baseDataPath + patientID + "/";
      std::vector<std::string> seriesDirs;

      for (const auto &entry : fs::directory_iterator(patientPath)) {
        if (entry.is_directory()) {
          seriesDirs.push_back(entry.path().string() + "/");
        }
      }

      if (seriesDirs.empty()) {
        throw std::runtime_error("No series directories found for patient: " +
                                 patientID);
      }

      // Use the first series directory found (usually there's only one)
      std::string seriesPath = seriesDirs[0];
      std::cout << "Using series directory: " << seriesPath << std::endl;

      std::vector<std::pair<std::string, int>> fileNumberPairs;

      for (const auto &entry : fs::directory_iterator(seriesPath)) {
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

      std::cout << "Found " << dicomFiles.size() << " DICOM files for patient "
                << patientID << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error loading DICOM files for patient " << patientID << ": "
                << e.what() << std::endl;
      throw;
    }
  }

  void processPatient(const std::string &patientID,
                      size_t batchSize = DEFAULT_BATCH_SIZE) {
    try {
      std::cout << "\n=== Processing Patient: " << patientID
                << " using Parallel Processing ===\n"
                << std::endl;

      // Setup output directory for this patient
      setupOutputDirectory(patientID);

      // Load DICOM files for this patient
      loadDICOMFilesForPatient(patientID);

      int successCount = 0;
      std::cout << "Found " << dicomFiles.size()
                << " images to process for patient " << patientID << std::endl;
      std::cout << "Using " << omp_get_max_threads() << " threads\n"
                << std::endl;

      // Process images in batches
      for (size_t batchStart = 0; batchStart < dicomFiles.size();
           batchStart += batchSize) {
        size_t currentBatchSize =
            std::min(batchSize, dicomFiles.size() - batchStart);
        std::vector<ProcessedImageData> batchResults(currentBatchSize);

#pragma omp parallel for schedule(auto) reduction(+ : successCount)
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

      std::cout << "\nPatient " << patientID
                << " completed. Successfully processed " << successCount << "/"
                << dicomFiles.size() << " images." << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error processing patient " << patientID << ": " << e.what()
                << std::endl;
      // Don't throw here - allow processing of other patients to continue
    }
  }

  void processAllPatients(size_t batchSize = DEFAULT_BATCH_SIZE) {
    std::cout << "\n=== Starting Parallel Processing for All Patients ===\n"
              << std::endl;

    // Find all patient directories
    std::vector<std::string> patientDirs = findAllPatientDirectories();

    if (patientDirs.empty()) {
      std::cout << "No patient directories found. Exiting." << std::endl;
      return;
    }

    // Process each patient directory
    int successfulPatients = 0;
    for (const auto &patientID : patientDirs) {
      try {
        processPatient(patientID, batchSize);
        successfulPatients++;
      } catch (const std::exception &e) {
        std::cerr << "Failed to process patient " << patientID
                  << ". Moving to next patient." << std::endl;
      }
    }

    std::cout << "\n=== All Processing Completed ===\n" << std::endl;
    std::cout << "Successfully processed " << successfulPatients << "/"
              << patientDirs.size() << " patients." << std::endl;
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

    // Over Subscribing OS-Level Worker Threads
    omp_set_num_threads(64);

    OptimizedParallelProcessor processor;
    processor.processAllPatients();

  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
