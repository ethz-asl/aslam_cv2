cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_calibration)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
add_definitions(-std=c++11)

set(HEADERS
  include/aslam/calibration/target-base.h
)

set(SOURCES
  src/focallength-initializers.cc
  src/target-aprilgrid.cc
  src/target-base.cc
)

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
add_doxygen(NOT_AUTOMATIC)

##########
# GTESTS #
##########
catkin_add_gtest(test_init_intrinsics test/test-init-intrinsics.cc)
target_link_libraries(test_init_intrinsics ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
