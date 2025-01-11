cmake_minimum_required(VERSION 3.16)
project(BrainTumorDetection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find FAST package
if(WIN32)
    set(FAST_DIR "C:/Program Files/FAST/fast/cmake/" CACHE PATH "Path to FAST installation")
else()
    set(FAST_DIR "/opt/fast/cmake/" CACHE PATH "Path to FAST installation")
endif()
find_package(FAST REQUIRED)

# Add executable
add_executable(BrainTumorDetection 
    src/main.cpp
)

# Link against FAST libraries
target_link_libraries(BrainTumorDetection PRIVATE FAST::FAST)

# Include directories
target_include_directories(BrainTumorDetection PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

