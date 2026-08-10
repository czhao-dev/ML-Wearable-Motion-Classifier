cmake_minimum_required(VERSION 3.21)

project(sdl
    VERSION 0.1.0
    DESCRIPTION "High-throughput synthetic-data training dataloader"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SDL_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(SDL_BUILD_BENCH "Build Google Benchmark harness" ON)
option(SDL_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(SDL_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(SDL_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(SDL_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)

add_library(sdl_warnings INTERFACE)
target_compile_options(sdl_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
)

add_library(sdl_sanitizers INTERFACE)
if(SDL_ENABLE_ASAN)
    target_compile_options(sdl_sanitizers INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(sdl_sanitizers INTERFACE -fsanitize=address)
endif()
if(SDL_ENABLE_UBSAN)
    target_compile_options(sdl_sanitizers INTERFACE -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(sdl_sanitizers INTERFACE -fsanitize=undefined)
endif()
if(SDL_ENABLE_TSAN)
    target_compile_options(sdl_sanitizers INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(sdl_sanitizers INTERFACE -fsanitize=thread)
endif()
if(SDL_ENABLE_COVERAGE)
    target_compile_options(sdl_sanitizers INTERFACE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(sdl_sanitizers INTERFACE -fprofile-instr-generate -fcoverage-mapping)
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)

if(SDL_BUILD_TESTS)
    enable_testing()
    include(cmake/GoogleTest.cmake)
    add_subdirectory(tests)
endif()

if(SDL_BUILD_BENCH)
    include(cmake/GoogleBenchmark.cmake)
    add_subdirectory(bench)
endif()
