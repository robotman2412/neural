
cmake_minimum_required(VERSION 3.10.0)

project(neural C)

add_executable(neural
    src/main.c
    src/neural.c
)

target_include_directories(neural PRIVATE include)
target_link_libraries(neural PRIVATE m)
