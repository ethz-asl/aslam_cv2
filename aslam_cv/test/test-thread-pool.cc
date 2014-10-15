#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/thread-pool.h>

int increment(int a){
  std::chrono::milliseconds dura(a);
  std::this_thread::sleep_for(dura);
  ++a;
  return a;
}

TEST(ThreadPoolTests, testBasic) {
  aslam::ThreadPool pool(2);
  auto job1 = pool.enqueue(&increment, 500);
  auto job2 = pool.enqueue(&increment, 400);
  auto job3 = pool.enqueue(&increment, 300);
  auto job4 = pool.enqueue(&increment, 200);
  auto job5 = pool.enqueue(&increment, 100);

  EXPECT_EQ(501, job1.get());
  EXPECT_EQ(401, job2.get());
  EXPECT_EQ(301, job3.get());
  EXPECT_EQ(201, job4.get());
  EXPECT_EQ(101, job5.get());
}

ASLAM_UNITTEST_ENTRYPOINT
