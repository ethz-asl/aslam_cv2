#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <aslam/common/entrypoint.h>
#include <aslam/common/eigen-helpers.h>
#include <aslam/common/opencv-predicates.h>
#include <aslam/common/channel-definitions.h>
#include <aslam/common/channel-serialization.h>

template <typename SCALAR>
class CvMatSerializationTest : public ::testing::Test {
 public:
  typedef SCALAR Scalar;
  static const int num_rows = 15;
  static const int num_cols = 20;
  static const int num_channels = 4;

  Scalar getValue(int row, int col, int channel) {
    return row + num_cols * col + num_cols * num_rows * channel;
  }

  void fill(cv::Mat * image) {
    CHECK_NOTNULL(image);
    for(int r = 0; r < image->rows; ++r) {
      for( int c = 0; c < image->cols; ++c) {
        for( int ch = 0; ch < image->channels(); ++ch) {
          (&image->at<Scalar>(r,c))[ch] = getValue(r, c, ch);
        }
      }
    }
  }

  virtual void SetUp() {
    for(int i = 0; i < num_channels; ++i) {
      int type = CV_MAKETYPE(cv::DataType<SCALAR>::type, i);
      // http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-create
      // Create should only allocate if necessary.
      imagesA_[i].create(num_rows, num_cols, type);
      fill(&(imagesA_[i]));
    }
  }
  cv::Mat imagesA_[num_channels];
  cv::Mat imagesB_[num_channels];
};

typedef ::testing::Types< int32_t, uint8_t,  int8_t, uint16_t, int16_t,
    float, double> TypeTests;
TYPED_TEST_CASE(CvMatSerializationTest, TypeTests);

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
  aslam::channels::Channel<MatrixType> value_a;
  aslam::channels::Channel<MatrixType> value_b;
};

#define MAKE_TYPE_LIST(Scalar) \
      Eigen::Matrix<Scalar, 20, 15>, \
      Eigen::Matrix<Scalar, Eigen::Dynamic, 10>, \
      Eigen::Matrix<Scalar, 25, Eigen::Dynamic>, \
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>

typedef ::testing::Types<MAKE_TYPE_LIST(double),
                         MAKE_TYPE_LIST(float),
                         MAKE_TYPE_LIST(int),
                         MAKE_TYPE_LIST(unsigned char)> DoubleTests;
TYPED_TEST_CASE(ChannelSerializationTest, DoubleTests);

TYPED_TEST(ChannelSerializationTest, SerializeDeserializeString) {
  aslam::internal::HeaderInformation header_info;
  std::string serialized_value;
  EXPECT_TRUE(this->value_a.serializeToString(&serialized_value));
  EXPECT_EQ(16u, header_info.size());
  ASSERT_EQ(header_info.size() + this->value_a.value_.rows() *
            this->value_a.value_.cols() *
            sizeof(typename TypeParam::Scalar), serialized_value.size());
  EXPECT_FALSE(aslam::common::MatricesEqual(this->value_a.value_,
                                            this->value_b.value_, 1e-4));
  EXPECT_TRUE(this->value_b.deSerializeFromString(serialized_value));
  EXPECT_TRUE(aslam::common::MatricesEqual(this->value_a.value_,
                                           this->value_b.value_, 1e-4));
}

TYPED_TEST(ChannelSerializationTest, SerializeDeserializeBuffer) {
  aslam::internal::HeaderInformation header_info;
  char* buffer;
  size_t size;
  EXPECT_TRUE(this->value_a.serializeToBuffer(&buffer, &size));
  EXPECT_EQ(16u, header_info.size());
  ASSERT_EQ(header_info.size() + this->value_a.value_.rows() *
            this->value_a.value_.cols() *
            sizeof(typename TypeParam::Scalar), size);
  EXPECT_FALSE(aslam::common::MatricesEqual(this->value_a.value_,
                                            this->value_b.value_, 1e-4));
  EXPECT_TRUE(this->value_b.deSerializeFromBuffer(buffer, size));
  EXPECT_TRUE(aslam::common::MatricesEqual(this->value_a.value_,
                                           this->value_b.value_, 1e-4));
}

TEST(ChannelSerialization, HeaderInfoSize) {
  aslam::internal::HeaderInformation header_info;
  header_info.cols = 12;
  header_info.rows = 10;
  header_info.depth = 4;
  header_info.channels = 13;
  EXPECT_EQ(16u, header_info.size());
  std::string header_serialized;
  header_serialized.resize(12u);
  header_info.serializeToBuffer(&header_serialized[0], 0);

  aslam::internal::HeaderInformation header_info2;
  header_info2.deSerializeFromBuffer(&header_serialized[0], 0);
  EXPECT_EQ(12, header_info2.cols);
  EXPECT_EQ(10, header_info2.rows);
  EXPECT_EQ(4, header_info2.depth);
  ASSERT_EQ(13, header_info2.channels);
}

TEST(ChannelSerialization, SerializeDeserializeNamedChannelFromString) {
  static const int numKeypoints = 5;
  aslam::channels::VISUAL_KEYPOINT_MEASUREMENTS keypoints_a;
  keypoints_a.value_.resize(Eigen::NoChange, numKeypoints);
  keypoints_a.value_.setRandom();
  std::string serialized_value;
  EXPECT_TRUE(keypoints_a.serializeToString(&serialized_value));

  aslam::internal::HeaderInformation header_info;
  ASSERT_EQ(serialized_value.size(), header_info.size() + 2 * numKeypoints *
            sizeof(aslam::channels::VISUAL_KEYPOINT_MEASUREMENTS::Type::Scalar));

  aslam::channels::VISUAL_KEYPOINT_MEASUREMENTS keypoints_b;

  EXPECT_FALSE(aslam::common::MatricesEqual(keypoints_a.value_,
                                            keypoints_b.value_, 1e-4));
  EXPECT_TRUE(keypoints_b.deSerializeFromString(serialized_value));
  EXPECT_TRUE(aslam::common::MatricesEqual(keypoints_a.value_,
                                           keypoints_b.value_, 1e-4));
}


TYPED_TEST(CvMatSerializationTest, SerializeDeserializeString) {
  for(int ch = 0; ch < this->num_channels; ++ch) {
    std::string serialized_value;
    EXPECT_TRUE(aslam::internal::serializeToString(this->imagesA_[ch], &serialized_value));
    EXPECT_FALSE(gtest_catkin::ImagesEqual(this->imagesA_[ch],
                                           this->imagesB_[ch], 1e-4));
    EXPECT_TRUE(aslam::internal::deSerializeFromString(serialized_value, &this->imagesB_[ch]));
    EXPECT_TRUE(gtest_catkin::ImagesEqual(this->imagesA_[ch],
                                          this->imagesB_[ch], 1e-4));
    typedef TypeParam Scalar;
    // Unit test the comparison function
    this->imagesB_[ch].template at<Scalar>(1,1) = this->imagesB_[ch].template at<Scalar>(1,1) + 1;
    EXPECT_FALSE(gtest_catkin::ImagesEqual(this->imagesA_[ch],
                                          this->imagesB_[ch], 1e-4));
  }
}

