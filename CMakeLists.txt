cmake_minimum_required(VERSION 3.10)

# Project name
project(resnet18_dfa_dnnl)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(<PATH_TO_oneDNN>/include)
include_directories(<PATH_TO_CNPY>)

# Link directories
link_directories(<PATH_TO_oneDNN>/build/src)
link_directories(<PATH_TO_CNPY>/build)

# Source files
set(SOURCE_FILES resnet18_dfa_dnnl.cc)

# Executable
add_executable(inference ${SOURCE_FILES})

# Link libraries
target_link_libraries(inference dnnl cnpy)

# create diretory 
make_directory(${CMAKE_SOURCE_DIR}/outputs)