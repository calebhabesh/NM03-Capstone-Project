#include <FAST/Importers/DICOMFileImporter.hpp>
#include <omp.h>
#include <vector>
#include <string>

using namespace fast;

// Function to load DICOM images using OpenMP
std::vector<std::shared_ptr<Image>> loadDICOMOpenMP(const std::vector<std::string>& directories) {
    std::vector<std::shared_ptr<Image>> images(directories.size());

    #pragma omp parallel for
    for (size_t i = 0; i < directories.size(); ++i) {
        auto importer = DICOMFileImporter::create();
        importer->setFilename(directories[i]);
        importer->update();
        images[i] = importer->getOutputData<Image>();
    }

    return images;
}
