cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_geometric_vision)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
set(HEADERS
  include/aslam/geometric_vision/match-outlier-rejection-twopt.h
)

set(SOURCES
  src/match-outlier-rejection-twopt.cc
)

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

##########
# EXPORT #
##########
cs_install()
cs_export()
