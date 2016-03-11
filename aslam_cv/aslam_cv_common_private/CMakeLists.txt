cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_common_private)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

##########
# GTESTS #
##########

catkin_add_gtest(test_occupancy_grid test/test-occupancy-grid.cc)
target_link_libraries(test_occupancy_grid ${catkin_LIBRARIES})

catkin_add_gtest(test_descriptor_utils test/test-descriptor-utils.cc)
target_link_libraries(test_descriptor_utils ${catkin_LIBRARIES})

##########
# EXPORT #
##########
cs_install()
cs_export()
