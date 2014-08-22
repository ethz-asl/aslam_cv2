cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)


cs_add_library(${PROJECT_NAME} src/camera.cc
                               src/distortion.cc
                               src/pinhole-camera.cc
                               src/fisheye-distortion.cc
                               src/camera-rig.cc
                               src/channel-serialization.cc
                               src/visual-frame.cc
                               src/visual-multi-frame.cc)

add_doxygen()

add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0 -std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

catkin_add_gtest(${PROJECT_NAME}_test_pinhole_camera test/test-pinhole-camera.cc)
target_link_libraries(${PROJECT_NAME}_test_pinhole_camera ${PROJECT_NAME})

catkin_add_gtest(test_fisheye_camera test/test-fisheye-camera.cc)
target_link_libraries(test_fisheye_camera ${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_test_serialization
  test/test-channel-serialization.cc
)
target_link_libraries(${PROJECT_NAME}_test_serialization ${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_test_channel
  test/test-channels.cc
)
target_link_libraries(${PROJECT_NAME}_test_channel ${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_test_frames
  test/test-visual-frame.cc
)
target_link_libraries(${PROJECT_NAME}_test_frames ${PROJECT_NAME})


cs_install()
cs_export()

