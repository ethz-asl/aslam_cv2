cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_python)

find_package(catkin_simple REQUIRED)
catkin_simple()

add_python_export_library(${PROJECT_NAME} python/aslam_cv
  src/module.cc
)

cs_install()
cs_export()

