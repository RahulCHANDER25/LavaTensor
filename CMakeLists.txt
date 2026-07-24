cmake_minimum_required(VERSION 3.20)
project(LavaTensor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Core LavaTensor library source files
set(LAVA_SOURCES
    lava/Tensor/Tensor.cpp
    lava/Tensor/TensorArray.cpp
)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compilation enabled using ${CMAKE_CUDA_COMPILER}")
    add_compile_definitions(LAVA_HAS_CUDA)
    set(LAVA_CUDA_SOURCES
        lava/Tensor/CudaDevice.cu
    )
else()
    message(STATUS "CUDA compiler not found. Building CPU-only LavaTensor.")
    set(LAVA_CUDA_SOURCES "")
endif()

add_library(lava STATIC ${LAVA_SOURCES} ${LAVA_CUDA_SOURCES})

# Public include directory so that users of the library (tests, apps)
# can include headers as <Tensor/Tensor.hpp> or <nn/Linear.hpp>
target_include_directories(lava PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lava)

# Enable chess project sub-directory
option(BUILD_CHESS "Build the chess project applications" OFF)
if(BUILD_CHESS)
    add_subdirectory(projects/chess)
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(playground src/main.cpp)
    target_link_libraries(playground PRIVATE lava)
endif()

# Enable testing sub-directory
option(BUILD_TESTS "Build library unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
