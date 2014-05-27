#include <aslam/common/entrypoint.h>
#include <aslam/common/eigen-helpers.h>
#include <aslam/common/channel-definitions.h>
#include <aslam/common/channel-serialization.h>

#define RUN_EIGEN_MATRIX_CHANNEL_DESERIALIZATION_TEST_IMPL(TEST_NAME, TYPE) \
  TEST(ChannelSerialization, TEST_NAME) {                             \
    typedef GET_TYPE(TYPE) MatrixType;                                \
    aslam::Channel<MatrixType> value_a;                               \
    static const int num_rows = 15;                                   \
    static const int num_cols = 20;                                   \
    if (MatrixType::RowsAtCompileTime == Eigen::Dynamic &&            \
        MatrixType::ColsAtCompileTime == Eigen::Dynamic) {            \
      value_a.value_.resize(num_rows, num_cols);                      \
    } else if (MatrixType::RowsAtCompileTime == Eigen::Dynamic &&     \
        MatrixType::ColsAtCompileTime != Eigen::Dynamic) {            \
      value_a.value_.resize(num_rows, Eigen::NoChange);               \
    }  else if (MatrixType::RowsAtCompileTime != Eigen::Dynamic &&    \
        MatrixType::ColsAtCompileTime == Eigen::Dynamic) {            \
      value_a.value_.resize(Eigen::NoChange, num_cols);               \
    }                                                                 \
    value_a.value_.setRandom();                                       \
    aslam::internal::HeaderInformation header_info;                   \
    std::string serialized_value;                                     \
    EXPECT_TRUE(value_a.serializeToString(&serialized_value));        \
    EXPECT_EQ(12u, header_info.size());                               \
    ASSERT_EQ(header_info.size() + value_a.value_.rows() * value_a.value_.cols() *    \
        sizeof(MatrixType::Scalar), serialized_value.size());                         \
    aslam::Channel<MatrixType> value_b;                                               \
    EXPECT_FALSE(aslam::common::MatricesEqual(value_a.value_, value_b.value_, 1e-4)); \
    EXPECT_TRUE(value_b.deSerializeFromString(serialized_value));                     \
    EXPECT_TRUE(aslam::common::MatricesEqual(value_a.value_, value_b.value_, 1e-4));  \
}

#define RUN_EIGEN_MATRIX_CHANNEL_DESERIALIZATION_TEST(x, ...) \
  RUN_EIGEN_MATRIX_CHANNEL_DESERIALIZATION_TEST_IMPL(x, (__VA_ARGS__))

TEST(ChannelSerialization, HeaderInfoSize) {
  aslam::internal::HeaderInformation header_info;
  header_info.cols = 12;
  header_info.rows = 10;
  header_info.type = 4;
  EXPECT_EQ(header_info.size(), 12u);
  std::string header_serialized;
  header_serialized.resize(12u);
  header_info.serializeToString(&header_serialized[0], 0);

  aslam::internal::HeaderInformation header_info2;
  header_info2.deSerializeFromString(&header_serialized[0], 0);
  EXPECT_EQ(header_info2.cols, 12);
  EXPECT_EQ(header_info2.rows, 10);
  EXPECT_EQ(header_info2.type, 4);
}

RUN_EIGEN_MATRIX_CHANNEL_DESERIALIZATION_TEST(FixedSize, Eigen::Matrix<double, 20, 15>);
RUN_EIGEN_MATRIX_CHANNEL_DESERIALIZATION_TEST(DynamicFixedSize, Eigen::Matrix<double, Eigen::Dynamic, 10>);
RUN_EIGEN_MATRIX_CHANNEL_DESERIALIZATION_TEST(FixedDynamicSize, Eigen::Matrix<double, 25, Eigen::Dynamic>);
RUN_EIGEN_MATRIX_CHANNEL_DESERIALIZATION_TEST(DynamicSize, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>);

TEST(ChannelSerialization, SerializeDeserializeNamedChannelFromString) {
  static const int numKeypoints = 5;
  aslam::channels::VISUAL_KEYPOINTS keypoints_a;
  keypoints_a.value_.resize(Eigen::NoChange, numKeypoints);
  keypoints_a.value_.setRandom();
  std::string serialized_value;
  EXPECT_TRUE(keypoints_a.serializeToString(&serialized_value));

  aslam::internal::HeaderInformation header_info;
  ASSERT_EQ(serialized_value.size(), header_info.size() + 2 * numKeypoints *
            sizeof(aslam::channels::VISUAL_KEYPOINTS::Type::Scalar));

  aslam::channels::VISUAL_KEYPOINTS keypoints_b;

  EXPECT_FALSE(aslam::common::MatricesEqual(keypoints_a.value_,
                                            keypoints_b.value_, 1e-4));
  EXPECT_TRUE(keypoints_b.deSerializeFromString(serialized_value));
  EXPECT_TRUE(aslam::common::MatricesEqual(keypoints_a.value_,
                                           keypoints_b.value_, 1e-4));
}

ASLAM_UNITTEST_ENTRYPOINT
