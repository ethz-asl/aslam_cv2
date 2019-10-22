include(${CMAKE_CURRENT_LIST_DIR}/detect_simd.cmake)

add_definitions(-std=c++11)

set(ENABLE_TIMING FALSE CACHE BOOL "Set to TRUE to enable timing")
message(STATUS "Timers enabled? ${ENABLE_TIMING}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_TIMING=${ENABLE_TIMING}")

set(ENABLE_STATISTICS FALSE CACHE BOOL "Set to TRUE to enable statistics")
message(STATUS "Statistics enabled? ${ENABLE_STATISTICS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_STATISTICS=${ENABLE_STATISTICS}")
