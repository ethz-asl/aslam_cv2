cmake_minimum_required(VERSION 2.8.3)
project(aslam_frames)

find_package(catkin_simple REQUIRED)
catkin_simple()

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} src/VisualFrame.cc)

catkin_add_gtest(${PROJECT_NAME}_test_frames
  test/testVisualFrame.cpp
)
target_link_libraries(${PROJECT_NAME}_test_frames ${PROJECT_NAME} pthread)

cs_install()
cs_export()
