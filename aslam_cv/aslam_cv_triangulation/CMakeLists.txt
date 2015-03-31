cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_triangulation)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} 
                               src/triangulation.cc
)

# TODO(tcies) reenable once sm timing fixed in tango
# cs_add_executable(triangulation-benchmark benchmark/triangulation-benchmark.cc)
# target_link_libraries(triangulation-benchmark ${PROJECT_NAME} gtest pthread)

add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
if(APPLE)
  add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
endif()
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

##########
# GTESTS #
##########
catkin_add_gtest(test_triangulation
    test/test-triangulation.cc
)
target_link_libraries(test_triangulation ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export()
