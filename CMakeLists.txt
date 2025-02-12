cmake_minimum_required(VERSION 3.1)

project("Brain Tumor Segmentation")

# For clangd lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(FAST REQUIRED)

include(${FAST_USE_FILE})

# Make executable for sequential code
add_executable(img_processing_sequential src/sequential/main_sequential.cpp)
add_dependencies(img_processing_sequential fast_copy)
target_link_libraries(img_processing_sequential  ${FAST_LIBRARIES})
target_include_directories(img_processing_sequential PRIVATE ${CMAKE_SOURCE_DIR}/src/include)


# Make executable for parallel code
add_executable(img_processing_parallel src/parallel/main_parallel.cpp)
add_dependencies(img_processing_parallel fast_copy)
target_link_libraries(img_processing_parallel ${FAST_LIBRARIES})
target_include_directories(img_processing_parallel PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

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

