cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_matcher)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
set(HEADERS
  include/aslam/matcher/match.h
  include/aslam/matcher/match-visualization.h
  include/aslam/matcher/matching-engine.h
  include/aslam/matcher/matching-engine-exclusive.h
  include/aslam/matcher/matching-engine-greedy.h
  include/aslam/matcher/matching-engine-non-exclusive.h
  include/aslam/matcher/matching-problem.h
  include/aslam/matcher/matching-problem-frame-to-frame.h
)

set(SOURCES
  src/match.cc
  src/match-visualization.cc
  src/matching-problem-frame-to-frame.cc
)

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

##########
# GTESTS #
##########
catkin_add_gtest(test_matcher test/test-matcher.cc)
target_link_libraries(test_matcher ${PROJECT_NAME}) 

catkin_add_gtest(test_matcher_non_exclusive test/test-matcher-non-exclusive.cc)
target_link_libraries(test_matcher_non_exclusive ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
