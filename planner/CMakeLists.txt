cmake_minimum_required(VERSION 2.8.12.2)
project(uncertain_terrain)

set(CFLAGS
    -std=c++11
    -g
)

add_compile_options(${CFLAGS})

include_directories(
    include
)

set(${PROJECT_NAME}_SOURCES
    2Dgridsearch.cpp
    config.cpp
    helpers.cpp
    planner.cpp
    environment_nav2D.cpp
    adplanner.cpp
    environment_navxythetalat.cpp 
    heap.cpp
    mdp.cpp
    utils.cpp
)

# Test
add_executable(test_planner test_main.cpp ${${PROJECT_NAME}_SOURCES})