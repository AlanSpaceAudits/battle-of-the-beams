cmake_minimum_required(VERSION 3.5)
project(GRwave Fortran)

add_executable(grwave.bin grwave.for)
target_compile_options(grwave.bin PRIVATE -w)
