# CMakeList.txt : CMake project for Connect4Pytorch, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (Connect4Pytorch "Connect4Pytorch.cpp" "Connect4Pytorch.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Connect4Pytorch PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
