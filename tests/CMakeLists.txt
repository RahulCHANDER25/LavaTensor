cmake_minimum_required(VERSION 3.20)
project(LavaTests LANGUAGES CXX)

# Test runner executable
add_executable(lava_tests
    main.cpp
)
target_link_libraries(lava_tests PRIVATE lava)

# Define test runner target for CTest validation
add_test(NAME CoreTests COMMAND lava_tests)
