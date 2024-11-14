cmake_minimum_required(VERSION 3.28)
project(int)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(int
        mtx_vec_main.c
)

target_link_libraries(int PRIVATE MPI::MPI_CXX)