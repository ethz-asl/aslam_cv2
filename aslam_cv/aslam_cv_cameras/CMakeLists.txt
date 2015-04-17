cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_cameras)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
set(HEADERS
  include/aslam/cameras/camera.h
  include/aslam/cameras/camera-factory.h
  include/aslam/cameras/camera-inl.h
  include/aslam/cameras/camera-pinhole.h
  include/aslam/cameras/camera-pinhole-inl.h
  include/aslam/cameras/camera-unified-projection.h
  include/aslam/cameras/distortion.h
  include/aslam/cameras/distortion-equidistant.h
  include/aslam/cameras/distortion-fisheye.h
  include/aslam/cameras/distortion-fisheye-inl.h
  include/aslam/cameras/distortion-inl.h
  include/aslam/cameras/distortion-null.h
  include/aslam/cameras/distortion-radtan.h
  include/aslam/cameras/ncamera.h
  include/aslam/cameras/yaml/camera-yaml-serialization.h
  include/aslam/cameras/yaml/ncamera-yaml-serialization.h
)

set(SOURCES
  src/camera.cc
  src/camera-factory.cc
  src/camera-pinhole.cc
  src/camera-unified-projection.cc
  src/camera-yaml-serialization.cc
  src/distortion.cc
  src/distortion-equidistant.cc
  src/distortion-fisheye.cc
  src/distortion-radtan.cc
  src/ncamera.cc
  src/ncamera-yaml-serialization.cc
)

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

##########
# GTESTS #
##########
catkin_add_gtest(test_cameras test/test-cameras.cc)
target_link_libraries(test_cameras ${PROJECT_NAME}) 

catkin_add_gtest(test_distortions test/test-distortions.cc)
target_link_libraries(test_distortions ${PROJECT_NAME}) 

catkin_add_gtest(test_ncamera test/test-ncamera.cc)
target_link_libraries(test_ncamera ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
