#ifndef ASLAM_CV2_ASLAM_CV_COMMON_TEST_READER_WRITER_MUTEX_FIXTURE_H_
#define ASLAM_CV2_ASLAM_CV_COMMON_TEST_READER_WRITER_MUTEX_FIXTURE_H_

#include <atomic>
#include <string>

#include <aslam/common/reader-writer-lock.h>
#include <gtest/gtest.h>

constexpr int kMagicNumber = 29845;
constexpr int kNumCycles = 1000;

namespace aslam {

class ReaderWriterMutexFixture : public ::testing::Test {
 private:
  int value_;
  std::atomic<int> num_writes_;
  std::atomic<int> num_upgrade_failures_;

 protected:
  virtual void SetUp();

  void reader();
  void writer();
  void delayedReader();
  void readerUpgrade();

  int value() {
    return value_;
  }
  int num_writes() {
    return num_writes_;
  }
  int num_upgrade_failures() {
    return num_upgrade_failures_;
  }

  ReaderWriterMutex value_mutex_;
};

}  // namespace aslam

#include "./reader_writer_mutex_fixture_inl.h"

#endif  // ASLAM_CV2_ASLAM_CV_COMMON_TEST_READER_WRITER_MUTEX_FIXTURE_H_
