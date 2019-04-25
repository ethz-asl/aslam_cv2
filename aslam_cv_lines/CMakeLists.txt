cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_lines)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME} src/dummy.cc)

catkin_add_gtest(test_lines test/test-lines.cc)
target_link_libraries(test_lines ${PROJECT_NAME})

cs_install()
cs_export()
