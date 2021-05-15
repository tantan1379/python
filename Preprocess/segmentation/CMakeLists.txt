cmake_minimum_required(VERSION 3.14)
project(segmentation)

set(CMAKE_CXX_STANDARD 14)

add_executable(segmentation main.cpp SGSmooth.cpp SGSmooth.hpp)

find_package(OpenCV REQUIRED)
target_link_libraries(segmentation ${OpenCV_LIBS})