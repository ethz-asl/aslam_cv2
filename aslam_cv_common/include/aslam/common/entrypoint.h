#ifndef GTEST_CATKIN_ENTRYPOINT_H_
#define GTEST_CATKIN_ENTRYPOINT_H_

#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#define ASLAM_UNITTEST_ENTRYPOINT\
  int main(int argc, char** argv) {\
  ::testing::InitGoogleTest(&argc, argv);\
  google::InitGoogleLogging(argv[0]);\
  google::ParseCommandLineFlags(&argc, &argv, false);\
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";\
  return RUN_ALL_TESTS();\
}

#endif  // GTEST_CATKIN_ENTRYPOINT_H_
