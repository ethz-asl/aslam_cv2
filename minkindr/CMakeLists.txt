cmake_minimum_required(VERSION 2.8.3)
project(minkindr)

find_package(catkin_simple REQUIRED)
catkin_simple()

cs_add_library(${PROJECT_NAME} 
  src/RotationQuaternion.cc
  src/QuatTransformation.cc
  src/AngleAxis.cc
)

# add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0)
catkin_add_gtest(${PROJECT_NAME}_tests 
  test/test_main.cc
  test/testRotation.cc
  test/testTransformation.cc
)

target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

cs_install()
cs_export()

