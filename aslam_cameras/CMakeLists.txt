cmake_minimum_required(VERSION 2.8.3)
project(aslam_cameras)

find_package(catkin_simple REQUIRED)
catkin_simple()

find_package(Boost REQUIRED COMPONENTS system serialization filesystem)

cs_add_library(${PROJECT_NAME} src/camera.cc
                               src/distortion.cc
                               src/pinhole-camera.cc)

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0)

catkin_add_gtest(${PROJECT_NAME}_tests test/test_pinhole-camera.cc)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

cs_install()
cs_export()
