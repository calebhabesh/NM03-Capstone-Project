// FAST Core includes
#include <FAST/Data/BoundingBox.hpp>
#include <FAST/Data/Color.hpp>
#include <FAST/Data/DataTypes.hpp>

// Visualization includes
#include <FAST/Visualization/MultiViewWindow.hpp>
#include <FAST/Visualization/RenderToImage/RenderToImage.hpp>
#include <FAST/Visualization/SegmentationRenderer/SegmentationRenderer.hpp>
#include <FAST/Visualization/SimpleWindow.hpp>

// Algorithm includes
#include <FAST/Algorithms/BinaryThresholding/BinaryThresholding.hpp>
#include <FAST/Algorithms/ImageCaster/ImageCaster.hpp>
#include <FAST/Algorithms/ImageSharpening/ImageSharpening.hpp>
#include <FAST/Algorithms/IntensityClipping/IntensityClipping.hpp>
#include <FAST/Algorithms/IntensityNormalization/IntensityNormalization.hpp>
#include <FAST/Algorithms/SeededRegionGrowing/SeededRegionGrowing.hpp>
#include <FAST/Algorithms/VectorMedianFilter/VectorMedianFilter.hpp>

// Morphology includes
#include <FAST/Algorithms/Morphology/Dilation.hpp>
#include <FAST/Algorithms/Morphology/Erosion.hpp>
#include <FAST/Algorithms/RegionProperties/RegionProperties.hpp>

// I/O includes
#include <FAST/Exporters/ImageExporter.hpp>
#include <FAST/Exporters/ImageFileExporter.hpp>
#include <FAST/Exporters/MetaImageExporter.hpp>
#include <FAST/Importers/DICOMFileImporter.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>

// System includes
#include <sys/types.h>
