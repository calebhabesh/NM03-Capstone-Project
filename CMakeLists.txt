cmake_minimum_required(VERSION 3.1)

project("Brain Tumor Segmentation")

# For clangd lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(FAST REQUIRED)

include(${FAST_USE_FILE})

# Make executable for sequential code
add_executable(brain_tumor_segmentation_sequential src/sequential/main_sequential.cpp)
add_dependencies(brain_tumor_segmentation fast_copy)
target_link_libraries(brain_tumor_segmentation_sequential ${FAST_LIBRARIES})

# Make executable for parallel code
add_executable(brain_tumor_segmentation_parallel src/parallel/main_parallel.cpp)
add_dependencies(brain_tumor_segmentation_parallel fast_copy)
target_link_libraries(brain_tumor_segmentation_parallel ${FAST_LIBRARIES})
