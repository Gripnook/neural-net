cmake_minimum_required (VERSION 3.1)
set (CMAKE_CXX_STANDARD 14)

project (neural-net)

add_executable (neural-net main.cpp neural-network.cpp)
