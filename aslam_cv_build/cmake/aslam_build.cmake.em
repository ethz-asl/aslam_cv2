macro(init_aslam_build)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wpedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")
  
  # Enable some clang specific options.
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wthread-safety -Wfor-loop-analysis")
  endif()
  message(STATUS "Setting aslam_cv_build settings.")
  
  find_package(catkin_simple REQUIRED)
  catkin_simple(ALL_DEPS_REQUIRED)
endmacro()
