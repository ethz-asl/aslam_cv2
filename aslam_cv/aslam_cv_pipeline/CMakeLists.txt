cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_pipeline)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} 
                               src/undistorter.cc
                               src/undistorter-mapped.cc
                               src/visual-npipeline.cc
                               src/visual-pipeline-brisk.cc
                               src/visual-pipeline-null.cc
                               src/visual-pipeline.cc
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
catkin_add_gtest(test_undistorters
    test/test-undistorters.cc
)
target_link_libraries(test_undistorters ${PROJECT_NAME}) 

catkin_add_gtest(test_visual-npipeline
    test/test-visual-npipeline.cc
)
target_link_libraries(test_visual-npipeline ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export()
