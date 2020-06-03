cmake_minimum_required(VERSION 3.10)

# compile using the c++11 standard and the g++ compiler
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER g++)

# find the external Eigen library
find_package(Eigen3 REQUIRED)

# include the header files in the project
include_directories(include)
include_directories(include/mgrit)
include_directories(include/neural_network)

# set the project name
project(Multigrid)

# add the src and tests directories to the project to be processed
add_subdirectory(src)
add_subdirectory(tests)