#include <FAST/Algorithms/NonLocalMeans/NonLocalMeans.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Visualization/DualViewWindow.hpp>
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>
#include <FAST/Visualization/SimpleWindow.hpp>

using namespace fast;

int main(int argc, char **argv) {
  auto importer = ImageFileImporter::create(
      Config::getTestDataPath() + "US/Heart/ApicalFourChamber/US-2D_0.mhd");

  // Set up the NonLocalMeans processing step and connect it to the importer
  auto filter = NonLocalMeans::create()->connect(importer);

  // Set a renderer and connect it to importer
  auto renderer = ImageRenderer::create()->connect(importer);

  // Set a renderer and connect it to the NonLocalMeans filter
  auto filterRenderer = ImageRenderer::create()->connect(filter);

  // Create a dual view window, add the renderers and start the
  // computation/rendering loop.
  DualViewWindow2D::create()
      ->connectLeft(renderer)
      ->connectRight(filterRenderer)
      ->run();
  return 0;
}
