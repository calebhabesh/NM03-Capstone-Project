cmake_minimum_required(VERSION 3.1)

project("Brain Tumor Segmentation")

# For clangd lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(FAST REQUIRED)

include(${FAST_USE_FILE})

add_executable(brain_tumor_segmentation src/main.cpp)
add_dependencies(brain_tumor_segmentation fast_copy)
target_link_libraries(brain_tumor_segmentation ${FAST_LIBRARIES})
