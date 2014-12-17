cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv)

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
                               src/channel-serialization.cc
                               src/distortion.cc
                               src/distortion-equidistant.cc
                               src/distortion-fisheye.cc
                               src/distortion-radtan.cc
                               src/feature-tracker-gyro.cc
                               src/feature-tracker.cc
                               src/matching-problem-frame-to-frame.cc
                               src/ncamera.cc
                               src/statistics.cc
                               src/thread-pool.cc
                               src/undistorter.cc
                               src/undistorter-mapped.cc
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

##########
# GTESTS #
##########
catkin_add_gtest(test_cameras
    test/test-cameras.cc
)
target_link_libraries(test_cameras ${PROJECT_NAME}) 

catkin_add_gtest(test_channel-serialization
    test/test-channel-serialization.cc
)
target_link_libraries(test_channel-serialization ${PROJECT_NAME}) 

catkin_add_gtest(test_channels
    test/test-channels.cc
)
target_link_libraries(test_channels ${PROJECT_NAME}) 

catkin_add_gtest(test_distortions
    test/test-distortions.cc
)
target_link_libraries(test_distortions ${PROJECT_NAME}) 

catkin_add_gtest(test_matcher
    test/test-matcher.cc
)
target_link_libraries(test_matcher ${PROJECT_NAME}) 

catkin_add_gtest(test_matcher_non_exclusive
    test/test-matcher-non-exclusive.cc
)
target_link_libraries(test_matcher_non_exclusive ${PROJECT_NAME}) 

catkin_add_gtest(test_triangulation
    test/test-triangulation.cc
)
target_link_libraries(test_triangulation ${PROJECT_NAME}) 

catkin_add_gtest(test_thread-pool
    test/test-thread-pool.cc
)
target_link_libraries(test_thread-pool ${PROJECT_NAME}) 

catkin_add_gtest(test_time
    test/test-time.cc
)
target_link_libraries(test_time ${PROJECT_NAME}) 

catkin_add_gtest(test_undistorters
    test/test-undistorters.cc
)
target_link_libraries(test_undistorters ${PROJECT_NAME}) 

catkin_add_gtest(test_visual-frame
    test/test-visual-frame.cc
)
target_link_libraries(test_visual-frame ${PROJECT_NAME}) 

catkin_add_gtest(test_visual-nframe
    test/test-visual-nframe.cc
)
target_link_libraries(test_visual-nframe ${PROJECT_NAME}) 

catkin_add_gtest(test_visual-npipeline
    test/test-visual-npipeline.cc
)
target_link_libraries(test_visual-npipeline ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export()
