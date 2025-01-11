#include <FAST/Tools/CommandLineParser.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>
#include <FAST/Visualization/SimpleWindow2D.hpp>

using namespace fast;

int main(int argc, char** argv) {
    Reporter::setGlobalReportMethod(Reporter::COUT);
    
    // Example of loading and displaying an image
    auto importer = ImageFileImporter::create();
    
    // TODO: Replace with actual path to your test image
    importer->setFilename("path/to/your/test/image.mhd");
    
    // Create renderer
    auto renderer = ImageRenderer::create()->connect(importer);
    
    // Create window and start rendering
    auto window = SimpleWindow2D::create()
        ->connect(renderer);
    window->run();

    return 0;
}

