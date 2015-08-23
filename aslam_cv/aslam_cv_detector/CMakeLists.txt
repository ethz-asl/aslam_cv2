cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_detector)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
set(HEADERS
  include/kaze/fed.h
  include/kaze/KAZE.h
  include/kaze/KAZEConfig.h
  include/kaze/nldiffusion_functions.h
  include/kaze/utils.h
)

set(SOURCES
  src/kaze/fed.cpp
  src/kaze/KAZE.cpp
  src/kaze/nldiffusion_functions.cpp
  src/kaze/utils.cpp
)

cs_add_library(${PROJECT_NAME} ${SOURCES})

add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

##########
# EXPORT #
##########
cs_install()
cs_export()
