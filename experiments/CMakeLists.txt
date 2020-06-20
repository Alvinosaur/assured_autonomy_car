cmake_minimum_required(VERSION 3.10)

# set the project name
project(test_cmake_mex)

find_package(Matlab)

# add the executable
# add_executable(test_cmake_mex hello.cpp)

matlab_add_mex(NAME test_cmake_mex SRC hello.cpp)