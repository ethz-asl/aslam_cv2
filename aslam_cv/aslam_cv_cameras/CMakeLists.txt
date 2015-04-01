cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_cameras)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} 
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
