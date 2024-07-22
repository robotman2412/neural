
cmake_minimum_required(VERSION 3.10.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS true)

project(neural C)

add_executable(neural
    src/gui.c
    src/main.c
    src/neural.c
    src/vecmath.c
)
target_include_directories(neural PRIVATE include)

find_package(SDL2 REQUIRED)
target_include_directories(neural PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(neural PRIVATE ${SDL2_LIBRARIES})

add_subdirectory(pax-gfx)
target_link_libraries(neural PRIVATE pax_graphics)

target_link_libraries(neural PRIVATE m)
add_compile_options(-march=native)
