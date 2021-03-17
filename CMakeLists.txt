cmake_minimum_required(VERSION 3.17)
project(torchscript_practice)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(torchscript_practice main.cpp)
target_link_libraries(torchscript_practice "${TORCH_LIBRARIES}")