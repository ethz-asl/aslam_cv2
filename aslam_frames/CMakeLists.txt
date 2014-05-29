cmake_minimum_required(VERSION 2.8.3)
project(aslam_frames)

find_package(catkin_simple REQUIRED)
catkin_simple()

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} src/VisualFrame.cc)

cs_install()
cs_export()
