cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_common)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} 
                               src/channel.cc
                               src/channel-serialization.cc
                               src/statistics.cc
                               src/thread-pool.cc
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
catkin_add_gtest(test_channel-serialization
    test/test-channel-serialization.cc
)
target_link_libraries(test_channel-serialization ${PROJECT_NAME}) 

catkin_add_gtest(test_channels
    test/test-channels.cc
)
target_link_libraries(test_channels ${PROJECT_NAME}) 

catkin_add_gtest(test_thread-pool
    test/test-thread-pool.cc
)
target_link_libraries(test_thread-pool ${PROJECT_NAME}) 

catkin_add_gtest(test_time
    test/test-time.cc
)
target_link_libraries(test_time ${PROJECT_NAME}) 

catkin_add_gtest(test_eigen-yaml-serialization
    test/test-eigen-yaml-serialization.cc
)
target_link_libraries(test_eigen-yaml-serialization ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export()
