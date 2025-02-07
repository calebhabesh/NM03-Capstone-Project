// Add the mountain of modules
#include "FAST/includes.hpp"

#include <sys/types.h>

using namespace fast;

int main(int argc, char **argv) {
  // Define the DICOM File Importer and point to T1C Brain Tumor Dataset .dcm
  // file for testing
  auto importer = DICOMFileImporter::create(
      Config::getTestDataPath() +
      "Brain-Tumor-Progression/PGBM-017/09-17-1997-RA FH MR RCBV "
      "OP-85753/16.000000-T1post-19554/1-14.dcm");

  // Very important for DICOM Importing
  // Load only one 2D slice image at a time, not the entire 3D volume
  // Otherwise will cause issues where filter interprets DICOM as 3D volume
  importer->setLoadSeries(false);

  // == Image Preprocessing Stage ==

  // 1. Intensity Normalization
  auto normalize = IntensityNormalization::create(0.5f, 2.5f, 0.0f, 10000.0f);
  normalize->connect(importer);
  // 2. Intensity Clipping
  auto clipping = IntensityClipping::create(0.68f, 4000.0f);
  clipping->connect(normalize);
  // 3. VMF Filter (Denoise, and preserves edges)
  auto medianfilter = VectorMedianFilter::create(5);
  medianfilter->connect(clipping);
  // 4. Sharpen (Sharpen edges)
  auto sharpen = ImageSharpening::create(2.0f, 0.5f, 9);
  sharpen->connect(medianfilter);

  // =============================================================

  // == Segmentation Stage ==
  // SeededRegionGrowing Segmentation
  // Create seeded region growing w/ similar intenisty range as thresholding
  auto regionGrowing = SeededRegionGrowing::create(
      0.74f, // minimum intensity (from your thresholding)
      0.91f, // maximum intensity (from your thresholding)
      std::vector<Vector3i>{
          // Grid of seed points in the right hemisphere (common tumor location)
          {300, 256, 0}, // Center-right
          {320, 256, 0}, // More right
          {340, 256, 0}, // Far right
          {300, 236, 0}, // Upper right
          {300, 276, 0}, // Lower right

          // Grid of seed points in the left hemisphere
          {212, 256, 0}, // Center-left
          {192, 256, 0}, // More left
          {172, 256, 0}, // Far left
          {212, 236, 0}, // Upper left
          {212, 276, 0}  // Lower left
      });
  regionGrowing->connect(sharpen);

  // Add additional seed points in a grid pattern
  for (int x = 150; x < 362; x += 30) {
    for (int y = 150; y < 362; y += 30) {
      regionGrowing->addSeedPoint(x, y);
    }
  }

  // Create color mapping for segmentation
  LabelColors labelColors;
  labelColors[1] = Color::Black(); // Map label 1 to black color

  // =============================================================

  // == Post-Processing and Analysis Stage ==
  // 1. Morphological operations to clean up segmentation
  auto erosion = Erosion::create(3); // Remove small artifacts
  erosion->connect(regionGrowing);

  auto dilation =
      Dilation::create(3); // Restore size while keeping cleaned edges
  dilation->connect(erosion);

  // 2. Calculate region properties
  auto regionProps = RegionProperties::create();
  regionProps->connect(dilation);

  // =============================================================

  // == Visualization Stage ==
  // Renders for original,prefilter,segmentation
  auto segmentationRenderer =
      SegmentationRenderer::create(
          labelColors, // Now passing the std::map directly
          0.6f,        // Segmentation opacity
          1.0f,        // Border opacity
          2            // Border radius
          )
          ->connect(regionGrowing);

  auto original = ImageRenderer::create()->connect(importer);
  auto prefilter = ImageRenderer::create()->connect(sharpen);

  // Create a dual view window, add the renderers and start the
  // computation/rendering loop.
  auto multiWindow =
      MultiViewWindow::create(3, Color::White(), 1300, 450, false);

  // Add renderers
  multiWindow->addRenderer(0, original);
  multiWindow->addRenderer(1, prefilter);
  multiWindow->addRenderer(2, segmentationRenderer);

  // Set window title
  multiWindow->setTitle("Medical Image Processing Stages");

  // Start the window
  multiWindow->run();
  return 0;
}
