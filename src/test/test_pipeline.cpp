// Add modules from includes.hpp
#include "FAST/includes.hpp"

#include <sys/types.h>

using namespace fast;

// == Export Stage Helper Function ==
void exportImages(
    const std::string &outputPath, std::shared_ptr<RenderToImage> renderToImage,
    const std::vector<std::pair<std::string, std::shared_ptr<Renderer>>>
        &renderPairs) {

  // Create output directory
  system(("mkdir -p " + outputPath + " && cd " + outputPath + " && rm -rf *")
             .c_str());

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
  importer->update();

  // Get image dimensions to adjust seed points accordingly
  auto importedImage = importer->getOutputData<Image>(0);
  if (!importedImage) {
    throw Exception("Failed to get imported image");
  }

  int width = importedImage->getWidth();
  int height = importedImage->getHeight();

  // 2. == Image Preprocessing Stage ==
  // 1. Intensity Normalization
  auto normalize = IntensityNormalization::create(0.5f, 2.5f, 0.0f, 10000.0f);
  normalize->connect(importer);
  normalize->update();

  // 2. Intensity Clipping
  auto clipping = IntensityClipping::create(0.68f, 4000.0f);
  clipping->connect(normalize);
  clipping->update();

  // 3. VMF Filter (Denoise, and preserves edges)
  auto medianfilter = VectorMedianFilter::create(
      7); // Updated filter size to 7 from main_sequential
  medianfilter->connect(clipping);
  medianfilter->update();

  // 4. Sharpen (Sharpen edges)
  auto sharpen = ImageSharpening::create(2.0f, 0.5f, 9);
  sharpen->connect(medianfilter);
  sharpen->update();

  // =============================================================

  // 3. == Segmentation Stage ==
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

  // SeededRegionGrowing Segmentation with adaptive seed points
  auto regionGrowing = SeededRegionGrowing::create(0.74f, 0.91f, seedPoints);
  regionGrowing->connect(sharpen);

  // Add additional seed points in a grid pattern
  for (int x = width / 4; x < width * 3 / 4; x += width / 10) {
    for (int y = height / 4; y < height * 3 / 4; y += height / 10) {
      regionGrowing->addSeedPoint(x, y);
    }
  }

  regionGrowing->update();

  // =============================================================

  // 4. == Post-Processing Stage ==
  // Cast to uint8 for morphology operations
  auto caster = ImageCaster::create(TYPE_UINT8);
  caster->connect(regionGrowing);
  caster->update();

  // Morphological operations to clean up segmentation
  auto erosion = Erosion::create(3);
  erosion->connect(caster);
  erosion->update();

  auto dilation = Dilation::create(3);
  dilation->connect(caster);
  dilation->update();

  // =============================================================

  // 5. == Visualization Stage ==
  LabelColors labelColors;
  labelColors[1] = Color::White();

  auto original = ImageRenderer::create()->connect(importer);
  auto prefilter = ImageRenderer::create()->connect(sharpen);

  auto segmentationRenderer =
      SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
          ->connect(regionGrowing);

  auto erosion_render = SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
                            ->connect(erosion);

  // This will be the final segegmented result that will get exported into /out
  auto dilation_render =
      SegmentationRenderer::create(labelColors, 0.6f, 1.0f, 2)
          ->connect(dilation);

  auto multiWindow =
      MultiViewWindow::create(5, Color::Black(), 2300, 450, false);

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
  auto renderToImage = RenderToImage::create(Color::Black(), 512, 512);

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
