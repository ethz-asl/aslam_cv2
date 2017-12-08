# Enable compiler optimizations.
add_definitions(-march=native -mtune=native -std=c++11)
execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH)
if (ARCH MATCHES "^(arm)")
  # Assume that neon is available.
  add_definitions(-mfpu=neon)
else()
  # Assume a processor with which supports Streaming SIMD Extensions in this
  # case.
  add_definitions(-mssse2 -mssse3)
endif()

set(ENABLE_TIMING FALSE CACHE BOOL "Set to TRUE to enable timing")
message(STATUS "Timers enabled? ${ENABLE_TIMING}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_TIMING=${ENABLE_TIMING}")

set(ENABLE_STATISTICS FALSE CACHE BOOL "Set to TRUE to enable statistics")
message(STATUS "Statistics enabled? ${ENABLE_STATISTICS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_STATISTICS=${ENABLE_STATISTICS}")
