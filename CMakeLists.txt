cmake_minimum_required(VERSION 3.2...4.2)
project(mlcpp VERSION 1.0
        DESCRIPTION "A C++ library for machine learning algorithms"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mlcpp INTERFACE)
target_include_directories(mlcpp INTERFACE include/)

enable_testing()
add_subdirectory(tests)
