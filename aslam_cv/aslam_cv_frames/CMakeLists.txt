cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} 
                               src/visual-frame.cc
                               src/visual-nframe.cc
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
catkin_add_gtest(test_visual-frame
    test/test-visual-frame.cc
)
target_link_libraries(test_visual-frame ${PROJECT_NAME}) 

catkin_add_gtest(test_visual-nframe
    test/test-visual-nframe.cc
)
target_link_libraries(test_visual-nframe ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export()
