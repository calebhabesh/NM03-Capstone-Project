#include "FAST/FAST_directives.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

using namespace fast;
namespace fs = std::filesystem;

class SequentialImageProcessor {
private:
  std::vector<std::string> dicomFiles;
  std::string baseDataPath;
  std::string patientPath;
  std::string outputBasePath;
  std::string currentOutputPath;

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

  void setupOutputDirectory(const std::string &patientID) {
    try {
      currentOutputPath = outputBasePath + "/" + patientID;
      if (system(("mkdir -p " + currentOutputPath + " && cd " +
                  currentOutputPath + " && rm -rf *")
                     .c_str()) != 0) {
        throw std::runtime_error("Failed to setup output directory: " +
                                 currentOutputPath);
      }
      std::cout << "Created clean output directory: " + currentOutputPath
                << std::endl;
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
      auto exporter = ImageFileExporter::create(currentOutputPath + "/" +
                                                baseName + "_original.jpg");
      exporter->connect(renderToImage->getOutputData<Image>(0));
      exporter->update();

      // Export final result
      renderToImage->removeAllRenderers();
      renderToImage->connect(dilationRenderer);
      renderToImage->update();
      exporter = ImageFileExporter::create(currentOutputPath + "/" + baseName +
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
      : outputBasePath(outputDir) {
    baseDataPath = Config::getTestDataPath() +
                   "Brain-Tumor-Progression/T1-Post-Combined-P001-P020/";

    // Create main output directory
    if (system(("mkdir -p " + outputBasePath).c_str()) != 0) {
      throw std::runtime_error("Failed to create base output directory: " +
                               outputBasePath);
    }
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

  void processPatient(const std::string &patientID) {
    try {
      std::cout << "\n=== Processing Patient: " << patientID << " ===\n"
                << std::endl;

      // Setup output directory for this patient
      setupOutputDirectory(patientID);

      // Load DICOM files for this patient
      loadDICOMFilesForPatient(patientID);

      // Process each image
      int successCount = 0;
      for (size_t i = 0; i < dicomFiles.size(); ++i) {
        try {
          processSingleImage(dicomFiles[i]);
          successCount++;
        } catch (const std::exception &e) {
          std::cerr << "Failed to process image " << (i + 1) << " for patient "
                    << patientID << ". Moving to next image." << std::endl;
        }
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

  void processAllPatients() {
    // Set reporting method for different types of messages
    Reporter::setGlobalReportMethod(Reporter::INFO,
                                    Reporter::NONE); // Disable INFO messages
    Reporter::setGlobalReportMethod(Reporter::WARNING,
                                    Reporter::COUT); // Keep warnings to console
    Reporter::setGlobalReportMethod(Reporter::ERROR,
                                    Reporter::COUT); // Keep errors to console

    std::cout << "\n=== Starting Sequential Processing for All Patients ===\n"
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
        processPatient(patientID);
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
    processor.processAllPatients();
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
