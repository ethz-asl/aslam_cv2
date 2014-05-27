#include <aslam/common/entrypoint.h>
#include <aslam/common/eigen-helpers.h>
#include <aslam/common/channel-definitions.h>
#include <aslam/common/channel-serialization.h>

TEST(ChannelSerialization, SerializeDeserializeFromString) {
  static const int numKeypoints = 200;
  aslam::channels::VISUAL_KEYPOINTS keypoints_a;
  keypoints_a.value_.setRandom(Eigen::NoChange, numKeypoints);
  std::string serialized_value;
  keypoints_a.serializeToString(&serialized_value);

  aslam::internal::HeaderInformation header_info;

  EXPECT_EQ(serialized_value.size(), header_info.size() + 2 * numKeypoints *
            sizeof(aslam::channels::VISUAL_KEYPOINTS::Type::Scalar));

  aslam::channels::VISUAL_KEYPOINTS keypoints_b;

  EXPECT_FALSE(aslam::common::MatricesEqual(keypoints_a.value_,
                                            keypoints_b.value_, 1e-4));
  keypoints_b.deSerializeFromString(serialized_value);
  EXPECT_TRUE(aslam::common::MatricesEqual(keypoints_a.value_,
                                           keypoints_b.value_, 1e-4));
}

ASLAM_UNITTEST_ENTRYPOINT
