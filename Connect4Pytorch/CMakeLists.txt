# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(Connect4Pytorch)

# Set the C++ standard (adjust based on your needs)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Path to LibTorch installation
set(Torch_DIR "C:/LibTorch/share/cmake/Torch")  # Adjust this to the correct path of your LibTorch installation
set(CMAKE_PREFIX_PATH "C:/LibTorch")  # Path to LibTorch

# Find the Torch package
find_package(Torch REQUIRED)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Define your source files (Add your cpp files here)
set(SOURCES
    Connect4Pytorch/Connect4Pytorch.cpp
    # Add more source files as needed
)

# Add an executable for your project
add_executable(Connect4Pytorch ${SOURCES})

# Link the Torch library
target_link_libraries(Connect4Pytorch "${TORCH_LIBRARIES}")

# Enable CUDA if you have it and want to use it (optional)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    find_package(CUDA REQUIRED)
    target_link_libraries(Connect4Pytorch ${CUDA_LIBRARIES})
endif()

# Ensure LibTorch is found
if (NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found! Make sure CMAKE_PREFIX_PATH and Torch_DIR are correctly set.")
endif()
