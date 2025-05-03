# Minimum required CMake version
cmake_minimum_required(VERSION 3.18)

# Project name
project(Connect4Pytorch)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Enable CUDA Support
find_package(CUDA REQUIRED)

# Include CUDA
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
add_definitions(-DUSE_CUDA)

# Path to LibTorch installation
set(Torch_DIR "C:/LibTorch/share/cmake/Torch")  # Adjust to your LibTorch path
set(CMAKE_PREFIX_PATH "C:/LibTorch")  # Path to LibTorch

# Find Torch package
find_package(Torch REQUIRED)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Define source files
set(SOURCES
    Connect4Pytorch/Connect4Pytorch.cpp  # Main file
    Connect4Pytorch/Board.cpp            # Board logic
    Connect4Pytorch/Connect4Algorithm.h  # Old AI logic
    Connect4Pytorch/DQNAgent.cpp         # Deep Q-Network AI
)

# Create executable for the project
add_executable(Connect4Pytorch ${SOURCES})

# Link Torch and CUDA libraries (Removed nvToolsExt)
target_link_libraries(Connect4Pytorch "${TORCH_LIBRARIES}")

# Ensure LibTorch is correctly found
if (NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found! Make sure CMAKE_PREFIX_PATH and Torch_DIR are correctly set.")
endif()

