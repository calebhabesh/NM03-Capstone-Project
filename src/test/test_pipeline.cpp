// Add modules from includes.hpp
#include "FAST/Visualization/RenderToImage/RenderToImage.hpp"
#include "FAST/includes.hpp"

#include <sys/types.h>

using namespace fast;

// == Export Stage Helper Function ==
void exportImages(
    const std::string &outputPath, std::shared_ptr<RenderToImage> renderToImage,
    const std::vector<std::pair<std::string, std::shared_ptr<Renderer>>>
        &renderPairs) {

  // Create output directory
  system(("rm -rf " + outputPath + " && mkdir -p " + outputPath).c_str());

  // Export each image
  for (const auto &[filename, renderer] : renderPairs) {
    renderToImage->removeAllRenderers();
    renderToImage->connect(renderer);
    renderToImage->update();

    auto exporter =
        ImageFileExporter::create(outputPath + "/" + filename + ".jpg");
    exporter->connect(renderToImage->getOutputData<Image>(0));
    exporter->update();
  }
}

int main(int argc, char **argv) {
  // Define the DICOM File Importer and point to T1C Brain Tumor Dataset .dcm
  // file for testing
  // 1. == Input/Import Stage ==
  auto importer = DICOMFileImporter::create(
      Config::getTestDataPath() +
      "Brain-Tumor-Progression/PGBM-017/09-17-1997-RA FH MR RCBV "
      "OP-85753/16.000000-T1post-19554/1-14.dcm");

  // Very important for DICOM Importing
  // Load only one 2D slice image at a time, not the entire 3D volume
  // Otherwise will cause issues where filter interprets DICOM as 3D volume
  importer->setLoadSeries(false);

  // 2. == Image Preprocessing Stage ==

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

  // 3. == Segmentation Stage ==
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

  // =============================================================

  // 4. == Post-Processing Stage ==
  // Cast to uint8 for morphology operations
  auto caster = ImageCaster::create(TYPE_UINT8);
  caster->connect(regionGrowing);

  // Morphological operations to clean up segmentation
  auto erosion = Erosion::create(3);
  erosion->connect(caster);

  auto dilation = Dilation::create(3);
  dilation->connect(caster);

  // =============================================================

  // 5. == Visualization Stage ==
  LabelColors labelColors;
  labelColors[1] = Color::Black();

  auto original = ImageRenderer::create()->connect(importer);
  auto prefilter = ImageRenderer::create()->connect(sharpen);

  auto segmentationRenderer =
      SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
          ->connect(regionGrowing); // Connect to dilation output

  auto erosion_render = SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
                            ->connect(erosion); // Connect to dilation output

  // This will be the final segegmented result that will get exported into /out
  auto dilation_render =
      SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
          ->connect(dilation); // Connect to dilation output

  auto multiWindow =
      MultiViewWindow::create(5, Color::White(), 2300, 450, false);

  multiWindow->addRenderer(0, original);
  multiWindow->addRenderer(1, prefilter);
  multiWindow->addRenderer(2, segmentationRenderer);
  multiWindow->addRenderer(3, erosion_render);
  multiWindow->addRenderer(4, dilation_render);

  multiWindow->setTitle("Medical Image Processing Stages");
  multiWindow->run();

  // =============================================================

  // 6. == Export Stage ==
  // create output directory, create if it does not exist
  auto renderToImage = RenderToImage::create(Color::White(), 512, 512);

  // Define export configurations
  std::vector<std::pair<std::string, std::shared_ptr<Renderer>>> renderPairs = {
      {"original_image", ImageRenderer::create()->connect(importer)},
      {"preprocessed_image", ImageRenderer::create()->connect(sharpen)},
      {"segmentation", SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
                           ->connect(regionGrowing)},
      {"erosion_result",
       SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
           ->connect(erosion)},
      {"final_dilated_result",
       SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
           ->connect(dilation)}};

  exportImages("../out-test", renderToImage, renderPairs);

  return 0;
}
