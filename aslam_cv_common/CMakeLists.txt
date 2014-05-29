cmake_minimum_required(VERSION 2.8.3)
project(aslam_common)

find_package(catkin_simple REQUIRED)
catkin_simple()

add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME} src/channel-serialization.cpp)

# Avoid clash with tr1::tuple: https://code.google.com/p/googletest/source/browse/trunk/README?r=589#257
add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0)

catkin_add_gtest(${PROJECT_NAME}_test_serialization
  test/testChannelSerialization.cpp
)
target_link_libraries(${PROJECT_NAME}_test_serialization ${PROJECT_NAME} pthread)

catkin_add_gtest(${PROJECT_NAME}_test_channel
  test/testChannels.cpp
)
target_link_libraries(${PROJECT_NAME}_test_channel ${PROJECT_NAME} pthread)

cs_install()
cs_export()
