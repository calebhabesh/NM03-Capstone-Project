#include <FAST/Tools/CommandLineParser.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Importers/DICOMFileImporter.hpp>
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>
#include <FAST/Visualization/SimpleWindow2D.hpp>
#include <FAST/Algorithms/ImageResampler/ImageResampler.hpp>
#include <future>
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <fstream>

using namespace fast;

// Function to load a DICOM image sequentially
std::shared_ptr<DICOMFileImporter> loadDICOMSequential(const std::string& directory) {
    auto importer = DICOMFileImporter::create();
    importer->setFilename(directory);
    importer->update();
    return importer;
}

// Function to load a DICOM image using parallelism
std::shared_ptr<DICOMFileImporter> loadDICOMParallel(const std::string& directory) {
    auto importer = DICOMFileImporter::create();
    importer->setFilename(directory);

    // Use async to load the image in parallel
    auto future = std::async(std::launch::async, [&importer]() {
        importer->update();
    });

    // Wait for the image to be loaded
    future.get();

    return importer;
}

// Function to preprocess MRI scans
std::shared_ptr<Image> preprocessMRI(std::shared_ptr<Image> image) {
    auto resampler = ImageResampler::create();
    resampler->setInputData(image);
    resampler->setOutputSpacing(1.0, 1.0, 1.0);
    resampler->update();
    return resampler->getOutputData<Image>();
}

// Function to collect core usage data
void collectCoreUsageData(const std::string& filename) {
    std::ofstream file(filename);
    file << "Time,CoreUsage\n";
    for (int i = 0; i < 100; ++i) {
        int numThreads = omp_get_num_threads();
        file << i << "," << numThreads << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    file.close();
}

int main(int argc, char** argv) {
    Reporter::setGlobalReportMethod(Reporter::COUT);

    // Example of loading and displaying a DICOM image
    // TODO: Replace with actual path to your DICOM directory
    std::string dicomDirectory = "path/to/your/DICOM/directory";

    // Start collecting core usage data in a separate thread
    std::thread coreUsageThread(collectCoreUsageData, "core_usage.csv");

    // Sequential loading
    auto start = std::chrono::high_resolution_clock::now();
    auto importerSeq = loadDICOMSequential(dicomDirectory);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSeq = end - start;
    std::cout << "Sequential loading time: " << durationSeq.count() << " seconds" << std::endl;

    // Parallel loading
    start = std::chrono::high_resolution_clock::now();
    auto importerPar = loadDICOMParallel(dicomDirectory);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationPar = end - start;
    std::cout << "Parallel loading time: " << durationPar.count() << " seconds" << std::endl;

    // Preprocess the image
    auto preprocessedImage = preprocessMRI(importerPar->getOutputData<Image>());

    // Create renderer
    auto renderer = ImageRenderer::create()->connect(preprocessedImage);

    // Create window and start rendering
    auto window = SimpleWindow2D::create()
        ->connect(renderer);
    window->run();

    // Wait for the core usage thread to finish
    coreUsageThread.join();

    return 0;
}