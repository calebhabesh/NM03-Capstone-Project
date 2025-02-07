cmake_minimum_required(VERSION 3.1)

project("Brain Tumor Segmentation")

# For clangd lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(FAST REQUIRED)

include(${FAST_USE_FILE})

# Make executable for sequential code
add_executable(brain_tumor_segmentation_sequential src/sequential/main_sequential.cpp)
add_dependencies(brain_tumor_segmentation_sequential fast_copy)
target_link_libraries(brain_tumor_segmentation_sequential ${FAST_LIBRARIES})
target_include_directories(brain_tumor_segmentation_sequential PRIVATE ${CMAKE_SOURCE_DIR}/src/include)


# Make executable for parallel code
add_executable(brain_tumor_segmentation_parallel src/parallel/main_parallel.cpp)
add_dependencies(brain_tumor_segmentation_parallel fast_copy)
target_link_libraries(brain_tumor_segmentation_parallel ${FAST_LIBRARIES})
target_include_directories(brain_tumor_segmentation_parallel PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Make executable for prototype/test code
add_executable(test_pipeline src/test/test_pipeline.cpp)
add_dependencies(test_pipeline fast_copy)
target_link_libraries(test_pipeline ${FAST_LIBRARIES})
target_include_directories(test_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Make executable for exported images test
add_executable(export_test src/test/export_test.cpp)
add_dependencies(export_test fast_copy)
target_link_libraries(export_test ${FAST_LIBRARIES})
target_include_directories(export_test PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

