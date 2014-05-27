#include <aslam/common/entrypoint.h>
#include <aslam/common/eigen-helpers.h>
#include <aslam/common/channel-definitions.h>
#include <aslam/common/channel-serialization.h>

template <typename TYPE>
class ChannelSerializationTest : public ::testing::Test {
 public:
  typedef TYPE MatrixType;
  static const int num_rows = 15;
  static const int num_cols = 20;

  virtual void SetUp() {
    if (MatrixType::RowsAtCompileTime == Eigen::Dynamic &&
        MatrixType::ColsAtCompileTime == Eigen::Dynamic) {
      value_a.value_.resize(num_rows, num_cols);
    } else if (MatrixType::RowsAtCompileTime == Eigen::Dynamic &&
        MatrixType::ColsAtCompileTime != Eigen::Dynamic) {
      value_a.value_.resize(num_rows, Eigen::NoChange);
    }  else if (MatrixType::RowsAtCompileTime != Eigen::Dynamic &&
        MatrixType::ColsAtCompileTime == Eigen::Dynamic) {
      value_a.value_.resize(Eigen::NoChange, num_cols);
    }
    value_a.value_.setRandom();
  }
  aslam::Channel<MatrixType> value_a;
  aslam::Channel<MatrixType> value_b;
};

#define MAKE_TYPE_LIST(Scalar) \
      Eigen::Matrix<Scalar, 20, 15>, \
      Eigen::Matrix<Scalar, Eigen::Dynamic, 10>, \
      Eigen::Matrix<Scalar, 25, Eigen::Dynamic>, \
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>

typedef ::testing::Types<MAKE_TYPE_LIST(double),
                         MAKE_TYPE_LIST(float),
                         MAKE_TYPE_LIST(int),
                         MAKE_TYPE_LIST(char)> DoubleTests;
TYPED_TEST_CASE(ChannelSerializationTest, DoubleTests);

TYPED_TEST(ChannelSerializationTest, SerializeDeserialize) {
  aslam::internal::HeaderInformation header_info;
  std::string serialized_value;
  EXPECT_TRUE(this->value_a.serializeToString(&serialized_value));
  EXPECT_EQ(12u, header_info.size());
  ASSERT_EQ(header_info.size() + this->value_a.value_.rows() *
            this->value_a.value_.cols() *
            sizeof(typename TypeParam::Scalar), serialized_value.size());
  EXPECT_FALSE(aslam::common::MatricesEqual(this->value_a.value_,
                                            this->value_b.value_, 1e-4));
  EXPECT_TRUE(this->value_b.deSerializeFromString(serialized_value));
  EXPECT_TRUE(aslam::common::MatricesEqual(this->value_a.value_,
                                           this->value_b.value_, 1e-4));
}

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
