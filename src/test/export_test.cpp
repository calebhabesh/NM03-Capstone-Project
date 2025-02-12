#include "FAST/includes.hpp"
#include <sys/types.h>

using namespace fast;

int main(int argc, char *argv[]) {
  // Now importing PNG instead of MHD
  auto importer = ImageFileImporter::create("../out/final_dilated_result.jpg");

  auto renderer = ImageRenderer::create()->connect(importer);

  SimpleWindow2D::create()->connect(renderer)->run();
  return 0;
}
