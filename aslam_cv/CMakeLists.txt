cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME} src/camera.cc
                               src/distortion.cc
                               src/pinhole-camera.cc
                               src/fisheye-distortion.cc
                               src/ncameras.cc
                               src/channel-serialization.cc
                               src/visual-frame.cc
                               src/visual-nframes.cc
                               src/thread-pool.cc
)

add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
if(APPLE)
  add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
endif()
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

catkin_add_gtest(${PROJECT_NAME}_tests 
    test/test-pinhole-camera.cc
    test/test-fisheye-camera.cc
    test/test-channel-serialization.cc
    test/test-channels.cc
    test/test-visual-frame.cc
    test/test-thread-pool.cc
    test/entry-point.cc
    )
target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

cs_install()
cs_export()

