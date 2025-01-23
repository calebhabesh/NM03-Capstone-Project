cmake_minimum_required(VERSION 3.1)

project("FAST example application")

# For clangd lsp
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(FAST REQUIRED)

include(${FAST_USE_FILE})

add_executable(fastExample src/main.cpp)
add_dependencies(fastExample fast_copy)
target_link_libraries(fastExample ${FAST_LIBRARIES})
