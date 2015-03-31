cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_matcher)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} 
                               src/match.cc
                               src/matching-problem-frame-to-frame.cc
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
catkin_add_gtest(test_matcher
    test/test-matcher.cc
)
target_link_libraries(test_matcher ${PROJECT_NAME}) 

catkin_add_gtest(test_matcher_non_exclusive
    test/test-matcher-non-exclusive.cc
)
target_link_libraries(test_matcher_non_exclusive ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
