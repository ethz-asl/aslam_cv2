# Detect the preprocessor directives which are set by the compiler.
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dM -E -x c /dev/null
                OUTPUT_VARIABLE PREPROCESSOR_DIRECTIVES)
execute_process(COMMAND cat /proc/cpuinfo COMMAND grep flags
                OUTPUT_VARIABLE CPU_FLAGS)

set(IS_SSE_ENABLED FALSE)
set(IS_NEON_ENABLED FALSE)
if (PREPROCESSOR_DIRECTIVES MATCHES "__SSE2__")
  if(${CPU_FLAGS} MATCHES "sse3")
    add_definitions(-mssse3)
    set(IS_SSE_ENABLED TRUE)
  else()
    message(FATAL_ERROR "SSE3 instruction set not available.")
  endif()
# For both armv7 and armv8, __ARM_NEON is used as preprocessor directive.
elseif (PREPROCESSOR_DIRECTIVES MATCHES "__ARM_ARCH 7")
  add_definitions(-mfpu=neon) # Needs to be set for armv7.
  set(IS_NEON_ENABLED TRUE)
elseif (PREPROCESSOR_DIRECTIVES MATCHES "__ARM_ARCH 8")
  set(IS_NEON_ENABLED TRUE)
endif()

unset(PREPROCESSOR_DIRECTIVES CACHE)
unset(CPU_FLAGS CACHE)
