cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_build)

find_package(catkin REQUIRED)

catkin_package(
  CATKIN_DEPENDS catkin
  CFG_EXTRAS aslam_build.cmake
)
