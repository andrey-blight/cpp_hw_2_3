cmake_minimum_required(VERSION 3.29)
project(untitled)

set(CMAKE_CXX_STANDARD 20)

set(TYPES
        "DOUBLE, FIXED(32,16)"
        CACHE ON "DOUBLE, FIXED(32,16)")
add_compile_definitions(-DTYPES=${TYPES})

add_executable(untitled
        test_main.cpp)
