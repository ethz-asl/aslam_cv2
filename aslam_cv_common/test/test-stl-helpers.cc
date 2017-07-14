#include <vector>

#include <aslam/common/entrypoint.h>
#include <aslam/common/stl-helpers.h>
#include <gtest/gtest.h>

using namespace aslam;

TEST(StlHelpers, EraseIndices) {
  std::vector<int> test_vector = {0, 1, 2, 3, 4, 5};
  std::vector<size_t> indices_to_remove = {2, 4};

  std::vector<int> result_vector =
      aslam::common::eraseIndicesFromVector(test_vector, indices_to_remove);
  ASSERT_EQ(result_vector.size(), test_vector.size() - indices_to_remove.size());

  std::vector<int> expected_result = {0, 1, 3, 5};
  ASSERT_EQ(result_vector.size(), expected_result.size());
  for (size_t idx = 0u; idx < result_vector.size(); ++idx) {
    EXPECT_EQ(expected_result[idx], result_vector[idx]);
  }
}

TEST(StlHelpers, EraseIndicesAligned) {
  Aligned<std::vector, Eigen::Vector3i> test_vector;
  test_vector.push_back(Eigen::Vector3i::Constant(0));
  test_vector.push_back(Eigen::Vector3i::Constant(1));
  test_vector.push_back(Eigen::Vector3i::Constant(2));
  test_vector.push_back(Eigen::Vector3i::Constant(3));
  std::vector<size_t> indices_to_remove = {2, 1};

  Aligned<std::vector, Eigen::Vector3i> result_vector =
      aslam::common::eraseIndicesFromVector(test_vector, indices_to_remove);
  ASSERT_EQ(result_vector.size(), test_vector.size() - indices_to_remove.size());

  std::vector<int> expected_result = {0, 3};
  ASSERT_EQ(result_vector.size(), expected_result.size());
  for (size_t idx = 0u; idx < result_vector.size(); ++idx) {
    EXPECT_EQ(expected_result[idx], result_vector[idx](0));
  }
}

TEST(StlHelpers, DrawRandom) {
  Aligned<std::vector, Eigen::Vector3i> test_vector;
  test_vector.push_back(Eigen::Vector3i::Constant(0));
  test_vector.push_back(Eigen::Vector3i::Constant(1));
  test_vector.push_back(Eigen::Vector3i::Constant(2));
  test_vector.push_back(Eigen::Vector3i::Constant(3));

  Aligned<std::vector, Eigen::Vector3i> output;

  const size_t kNum = 3;
  aslam::common::drawNRandomElements(kNum, test_vector, &output, true);
  EXPECT_EQ(output.size(), kNum);

  output.clear();
  aslam::common::drawNRandomElements(kNum, test_vector, &output, false);
  EXPECT_EQ(output.size(), kNum);

  output.clear();
  aslam::common::drawNRandomElements(kNum, test_vector, &output);
  EXPECT_EQ(output.size(), kNum);
}

TEST(StdHelpers, CountNestedListElements) {
  const size_t kArbitraryNumElementsOfList = 123u;
  Aligned<std::vector, int> eigen_list(kArbitraryNumElementsOfList);

  const size_t kArbitraryNumElementsOfNestedList = 93u;
  Aligned<std::vector, Aligned<std::vector, int>> eigen_nested_list(
      kArbitraryNumElementsOfNestedList, eigen_list);

  size_t num_elements = aslam::common::countNumberOfElementsInNestedList(
      eigen_nested_list);

  CHECK_EQ(num_elements,
           kArbitraryNumElementsOfList * kArbitraryNumElementsOfNestedList);

  std::vector<int> std_list(kArbitraryNumElementsOfList);
  std::vector<std::vector<int>> std_nested_list(
      kArbitraryNumElementsOfNestedList, std_list);

  num_elements = aslam::common::countNumberOfElementsInNestedList(
      std_nested_list);

  CHECK_EQ(num_elements,
           kArbitraryNumElementsOfList *  kArbitraryNumElementsOfNestedList);
}

ASLAM_UNITTEST_ENTRYPOINT
