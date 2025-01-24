#include "FAST/Data/DataTypes.hpp"
#include <FAST/Algorithms/GaussianSmoothing/GaussianSmoothing.hpp>
#include <FAST/Algorithms/ImageCaster/ImageCaster.hpp>
#include <FAST/Algorithms/ImageSharpening/ImageSharpening.hpp>
#include <FAST/Algorithms/NonLocalMeans/NonLocalMeans.hpp>
#include <FAST/Data/DataTypes.hpp>
#include <FAST/Importers/DICOMFileImporter.hpp>
#include <FAST/Visualization/DualViewWindow.hpp>
#include <FAST/Visualization/SimpleWindow.hpp>
#include <FAST/Visualization/SliceRenderer/SliceRenderer.hpp>
// #include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>

using namespace fast;

int main(int argc, char **argv) {
  // Define the DICOM File Importer and point to T1C Brain Tumor Dataset .dcm
  // images
  auto importer = DICOMFileImporter::create(
      Config::getTestDataPath() +
      "Brain-Tumor-Progression/PGBM-017/09-17-1997-RA FH MR RCBV "
      "OP-85753/16.000000-T1post-19554/1-14.dcm");

  // Set up the NonLocalMeans processing step and connect it to
  // the DICOM importer, currently has issues with data type
  auto filter = NonLocalMeans::create()->connect(importer);

  // GaussianSmoothing Class
  auto smoothing = GaussianSmoothing::create(2, 3)->connect(importer);

  // Extract a 2D slice from the original 3D image
  auto originalSliceRenderer =
      SliceRenderer::create(PlaneType::PLANE_Z, -1)->connect(importer);

  // Extract a 2D slice from the filtered image
  auto filteredSliceRenderer =
      SliceRenderer::create(PlaneType::PLANE_Z, -1)->connect(smoothing);

  // Create a dual view window, add the renderers and start the
  // computation/rendering loop.
  DualViewWindow2D::create()
      ->connectLeft(originalSliceRenderer)  // un-filtered image
      ->connectRight(filteredSliceRenderer) // filtered-image
      ->run();
  return 0;
}
