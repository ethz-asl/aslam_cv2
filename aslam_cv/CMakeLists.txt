cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_add_library(${PROJECT_NAME} 
                               src/camera-pinhole.cc
                               src/camera.cc
                               src/channel-serialization.cc
                               src/distortion-equidistant.cc
                               src/distortion-fisheye.cc
                               src/distortion-radtan.cc
                               src/distortion.cc
                               src/ncamera.cc
                               src/thread-pool.cc
                               src/undistorter-mapped.cc
                               src/undistorter.cc
                               src/visual-frame.cc
                               src/visual-nframe.cc
                               src/visual-npipeline.cc
                               src/visual-pipeline-brisk.cc
                               src/visual-pipeline-null.cc
                               src/visual-pipeline.cc
)

add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
if(APPLE)
  add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
endif()
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

catkin_add_gtest(${PROJECT_NAME}_tests 
    test/entry-point.cc
    test/test-cameras.cc
    test/test-channel-serialization.cc
    test/test-channels.cc
    test/test-distortions.cc  
    test/test-matcher.cc
    test/test-thread-pool.cc
    test/test-visual-frame.cc
    test/test-visual-npipeline.cc
    )
target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

cs_install()
cs_export()

