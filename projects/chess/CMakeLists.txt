cmake_minimum_required(VERSION 3.20)
project(LavaChess LANGUAGES CXX)

# Source files for the chess analyzer utility logic
set(CHESS_UTILS_SOURCES
    src/utils/FenConverter.cpp
    src/utils/NetworkLoader.cpp
)

# my_torch_generator executable
add_executable(my_torch_generator
    src/generator/main.cpp
)
target_link_libraries(my_torch_generator PRIVATE lava)
target_include_directories(my_torch_generator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/generator
)

# my_torch_analyzer executable
add_executable(my_torch_analyzer
    src/analyzer/main.cpp
    src/analyzer/training/chessTraining.cpp
    ${CHESS_UTILS_SOURCES}
)
target_link_libraries(my_torch_analyzer PRIVATE lava)
target_include_directories(my_torch_analyzer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/analyzer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/analyzer/training
)
