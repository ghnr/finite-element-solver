cmake_minimum_required(VERSION 3.6)
project(fe_solver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(SOURCE_FILES solver.cpp)

add_executable(fe_solver ${SOURCE_FILES})
include_directories("Eigen")


