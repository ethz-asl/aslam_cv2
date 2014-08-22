cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#find_package(Boost REQUIRED COMPONENTS system serialization filesystem)

add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0 -std=c++11)

cs_add_library(${PROJECT_NAME} src/camera.cc
                               src/distortion.cc
                               src/pinhole-camera.cc
                               src/fisheye-distortion.cc
                               src/camera-rig.cc
                               src/channel-serialization.cc)

#target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

catkin_add_gtest(${PROJECT_NAME}_tests test/test_pinhole-camera.cc)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

catkin_add_gtest(test_fisheye_camera test/test-fisheye-camera.cc)
target_link_libraries(test_fisheye_camera ${PROJECT_NAME})

catkin_add_gtest(${PROJECT_NAME}_test_serialization
  test/test-channel-serialization.cc
)
target_link_libraries(${PROJECT_NAME}_test_serialization ${PROJECT_NAME} pthread)

catkin_add_gtest(${PROJECT_NAME}_test_channel
  test/test-channels.cc
)
target_link_libraries(${PROJECT_NAME}_test_channel ${PROJECT_NAME} pthread)


cs_install()
cs_export()

