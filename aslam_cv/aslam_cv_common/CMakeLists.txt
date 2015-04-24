cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_common)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
set(HEADERS
  include/aslam/common/channel.h
  include/aslam/common/channel-declaration.h
  include/aslam/common/channel-definitions.h
  include/aslam/common/channel-serialization.h
  include/aslam/common/crtp-clone.h
  include/aslam/common/entrypoint.h
  include/aslam/common/feature-descriptor-ref.h
  include/aslam/common/feature-descriptor-ref-inl.h
  include/aslam/common/hamming.h
  include/aslam/common/hamming-inl.h
  include/aslam/common/hash-id.h
  include/aslam/common/lockable-container.h
  include/aslam/common/macros.h
  include/aslam/common/memory.h
  include/aslam/common/meta.h
  include/aslam/common/neon-helpers.h
  include/aslam/common/numdiff-jacobian-tester.h
  include/aslam/common/opencv-predicates.h
  include/aslam/common/pose-types.h
  include/aslam/common/predicates.h
  include/aslam/common/stl-helpers.h
  include/aslam/common/thread-pool.h
  include/aslam/common/time.h
  include/aslam/common/types.h
  include/aslam/common/undistort-helpers.h
  include/aslam/common/unique-id.h
  include/aslam/common/yaml-serialization.h
  include/aslam/common/yaml-serialization-eigen.h
  include/aslam/common/statistics/accumulator.h
  include/aslam/common/statistics/statistics.h
)

set(SOURCES
  src/channel.cc
  src/channel-serialization.cc
  src/hash-id.cc
  src/statistics.cc
  src/thread-pool.cc
)

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

##########
# GTESTS #
##########
catkin_add_gtest(test_channel-serialization test/test-channel-serialization.cc)
target_link_libraries(test_channel-serialization ${PROJECT_NAME}) 

catkin_add_gtest(test_channels test/test-channels.cc)
target_link_libraries(test_channels ${PROJECT_NAME}) 

catkin_add_gtest(test_thread-pool test/test-thread-pool.cc)
target_link_libraries(test_thread-pool ${PROJECT_NAME}) 

catkin_add_gtest(test_time test/test-time.cc)
target_link_libraries(test_time ${PROJECT_NAME}) 

catkin_add_gtest(test_hash-id test/test-hash-id.cc)
target_link_libraries(test_hash-id ${PROJECT_NAME})

catkin_add_gtest(test_eigen-yaml-serialization 
  test/test-eigen-yaml-serialization.cc
)
target_link_libraries(test_eigen-yaml-serialization ${PROJECT_NAME}) 

catkin_add_gtest(test_lockable_container 
  test/test-lockable-container.cc
)
target_link_libraries(test_lockable_container ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export(CFG_EXTRAS export_flags.cmake)
