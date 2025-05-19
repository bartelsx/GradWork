# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(Connect4Pytorch)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Path to LibTorch installation
set(Torch_DIR "C:/LibTorch/share/cmake/Torch")  # Adjust to your LibTorch path
set(CMAKE_PREFIX_PATH "C:/LibTorch")  # Path to LibTorch

# Find the Torch package
find_package(Torch REQUIRED)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Define your source files (keep using your existing structure)
set(SOURCES
    Connect4Pytorch/Connect4Pytorch.cpp  # Main file
    Connect4Pytorch/Board.cpp            # Board logic
    Connect4Pytorch/Connect4Algorithm.h        # Old AI
    Connect4Pytorch/DQNAgent.cpp         # New Deep Q-Network AI (to be implemented)
)

# Add an executable for your project
add_executable(Connect4Pytorch ${SOURCES})

# Link the Torch library
target_link_libraries(Connect4Pytorch "${TORCH_LIBRARIES}")

# Ensure LibTorch is found
if (NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found! Make sure CMAKE_PREFIX_PATH and Torch_DIR are correctly set.")
endif()
