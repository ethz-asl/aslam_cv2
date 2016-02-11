#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera-unified-projection.h>
#include <aslam/common/channel-declaration.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/opencv-predicates.h>
#include <aslam/cameras/camera.h>
#include <aslam/frames/visual-frame.h>

TEST(Frame, SetGetCamera) {
aslam::Camera::Ptr camera;
aslam::VisualFrame frame;
ASSERT_FALSE(static_cast<bool>(frame.getCameraGeometry()));
frame.setCameraGeometry(camera);
EXPECT_EQ(camera, frame.getCameraGeometry());
}

TEST(Frame, DeathGetElementFromOnUnsetData) {
aslam::VisualFrame frame;
EXPECT_DEATH(frame.getDescriptor(0), "^");
EXPECT_DEATH(frame.getKeypointMeasurement(0), "^");
EXPECT_DEATH(frame.getKeypointMeasurementUncertainty(0), "^");
EXPECT_DEATH(frame.getKeypointScale(0), "^");
EXPECT_DEATH(frame.getKeypointOrientation(0), "^");
}

TEST(Frame, DeathOnGetUnsetData) {
aslam::VisualFrame frame;
EXPECT_DEATH(frame.getDescriptors(), "^");
EXPECT_DEATH(frame.getKeypointMeasurements(), "^");
EXPECT_DEATH(frame.getKeypointMeasurementUncertainties(), "^");
EXPECT_DEATH(frame.getKeypointScales(), "^");
EXPECT_DEATH(frame.getKeypointOrientations(), "^");
EXPECT_DEATH(frame.getRawImage(), "^");
}

TEST(Frame, DeathOnGetMutableUnsetData) {
aslam::VisualFrame frame;
EXPECT_DEATH(frame.getDescriptorsMutable(), "^");
EXPECT_DEATH(frame.getKeypointMeasurementsMutable(), "^");
EXPECT_DEATH(frame.getKeypointMeasurementUncertaintiesMutable(), "^");
EXPECT_DEATH(frame.getKeypointScalesMutable(), "^");
EXPECT_DEATH(frame.getKeypointOrientationsMutable(), "^");
EXPECT_DEATH(frame.getRawImageMutable(), "^");
}

TEST(Frame, SetGetDescriptors) {
aslam::VisualFrame frame;
aslam::VisualFrame::DescriptorsT data;
data.resize(48, 10);
data.setRandom();
frame.setDescriptors(data);
const aslam::VisualFrame::DescriptorsT& data_2 =
    frame.getDescriptors();
EXPECT_TRUE(EIGEN_MATRIX_NEAR(data, data_2, 0));
EXPECT_EQ(&data_2, frame.getDescriptorsMutable());
for (int i = 0; i < data.cols(); ++i) {
  const unsigned char* data_ptr = frame.getDescriptor(i);
  EXPECT_EQ(&data_2.coeffRef(0, i), data_ptr);
}
}

TEST(Frame, SetGetKeypointMeasurements) {
aslam::VisualFrame frame;
Eigen::Matrix2Xd data;
data.resize(Eigen::NoChange, 10);
data.setRandom();
frame.setKeypointMeasurements(data);
const Eigen::Matrix2Xd& data_2 = frame.getKeypointMeasurements();
EXPECT_TRUE(EIGEN_MATRIX_NEAR(data, data_2, 1e-6));
EXPECT_EQ(&data_2, frame.getKeypointMeasurementsMutable());
for (int i = 0; i < data.cols(); ++i) {
  const Eigen::Vector2d& ref = frame.getKeypointMeasurement(i);
  const Eigen::Vector2d& should = data.block<2, 1>(0, i);
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(should, ref, 1e-6));
}
}

TEST(Frame, SetGetKeypointMeasurementUncertainties) {
aslam::VisualFrame frame;
Eigen::VectorXd data;
data.resize(10);
data.setRandom();
frame.setKeypointMeasurementUncertainties(data);
const Eigen::VectorXd& data_2 = frame.getKeypointMeasurementUncertainties();
EXPECT_TRUE(EIGEN_MATRIX_NEAR(data, data_2, 1e-6));
EXPECT_EQ(&data_2, frame.getKeypointMeasurementUncertaintiesMutable());
for (int i = 0; i < data.cols(); ++i) {
  double ref = frame.getKeypointMeasurementUncertainty(i);
  EXPECT_NEAR(data(i), ref, 1e-6);
}
}

TEST(Frame, SetGetKeypointOrientations) {
aslam::VisualFrame frame;
Eigen::VectorXd data;
data.resize(10);
data.setRandom();
frame.setKeypointOrientations(data);
const Eigen::VectorXd& data_2 = frame.getKeypointOrientations();
EXPECT_TRUE(EIGEN_MATRIX_NEAR(data, data_2, 1e-6));
EXPECT_EQ(&data_2, frame.getKeypointOrientationsMutable());
for (int i = 0; i < data.cols(); ++i) {
  double ref = frame.getKeypointOrientation(i);
  EXPECT_NEAR(data(i), ref, 1e-6);
}
}

TEST(Frame, SetGetKeypointScales) {
aslam::VisualFrame frame;
Eigen::VectorXd data;
data.resize(10);
data.setRandom();
frame.setKeypointScales(data);
const Eigen::VectorXd& data_2 = frame.getKeypointScales();
EXPECT_TRUE(EIGEN_MATRIX_NEAR(data, data_2, 1e-6));
EXPECT_EQ(&data_2, frame.getKeypointScalesMutable());
for (int i = 0; i < data.cols(); ++i) {
  double ref = frame.getKeypointScale(i);
  EXPECT_NEAR(data(i), ref, 1e-6);
}
}

TEST(Frame, NamedChannel) {
  aslam::VisualFrame frame;
  Eigen::VectorXd data;
  data.resize(10);
  data.setRandom();
  std::string channel_name = "test_channel";
  EXPECT_FALSE(frame.hasChannel(channel_name));
  EXPECT_DEATH(frame.getChannelData<Eigen::VectorXd>(channel_name), "^");

  frame.addChannel<Eigen::VectorXd>(channel_name);
  EXPECT_TRUE(frame.hasChannel(channel_name));
  frame.setChannelData(channel_name, data);

  const Eigen::VectorXd& data_2 =
      frame.getChannelData<Eigen::VectorXd>(channel_name);
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(data, data_2, 1e-6));
}

TEST(Frame, SetGetImage) {
  aslam::VisualFrame frame;
  cv::Mat data(10,10,CV_8SC3,uint8_t(7));

  frame.setRawImage(data);
  const cv::Mat& data_2 = frame.getRawImage();
  EXPECT_TRUE(gtest_catkin::ImagesEqual(data, data_2));
}

TEST(Frame, CopyConstructor) {
  aslam::Camera::Ptr camera = aslam::PinholeCamera::createTestCamera();
  aslam::VisualFrame frame;
  frame.setCameraGeometry(camera);

  // Set timestamps.
  constexpr int64_t kTimestamp = 100;
  frame.setTimestampNanoseconds(kTimestamp);

  // Set some random Data.
  constexpr size_t kNumRandomValues = 10;
  Eigen::Matrix2Xd keypoints = Eigen::Matrix2Xd::Random(2, kNumRandomValues);
  frame.setKeypointMeasurements(keypoints);
  Eigen::VectorXd uncertainties = Eigen::VectorXd::Random(1, kNumRandomValues);
  frame.setKeypointMeasurementUncertainties(uncertainties);
  Eigen::VectorXd orientations = Eigen::VectorXd::Random(1, kNumRandomValues);
  frame.setKeypointOrientations(orientations);
  Eigen::VectorXd scores = Eigen::VectorXd::Random(1, kNumRandomValues);
  frame.setKeypointScores(scores);
  Eigen::VectorXd scales = Eigen::VectorXd::Random(1, kNumRandomValues);
  frame.setKeypointScales(scales);
  aslam::VisualFrame::DescriptorsT descriptors =
      aslam::VisualFrame::DescriptorsT::Random(384, kNumRandomValues);
  frame.setDescriptors(descriptors);
  Eigen::VectorXi track_ids = Eigen::VectorXi::Random(1, 10);
  frame.setTrackIds(track_ids);

  // Set image.
  cv::Mat image = cv::Mat(3, 2, CV_8UC1);
  cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(255));
  frame.setRawImage(image);

  // Clone and compare.
  aslam::VisualFrame frame_cloned(frame);
  EXPECT_EQ(camera.get(), frame_cloned.getCameraGeometry().get());

  EXPECT_EQ(kTimestamp, frame_cloned.getTimestampNanoseconds());

  EIGEN_MATRIX_EQUAL(keypoints, frame_cloned.getKeypointMeasurements());
  EIGEN_MATRIX_EQUAL(uncertainties, frame_cloned.getKeypointMeasurementUncertainties());
  EIGEN_MATRIX_EQUAL(orientations, frame_cloned.getKeypointOrientations());
  EIGEN_MATRIX_EQUAL(scores, frame_cloned.getKeypointScores());
  EIGEN_MATRIX_EQUAL(scales, frame_cloned.getKeypointScales());
  EIGEN_MATRIX_EQUAL(descriptors, frame_cloned.getDescriptors());
  EIGEN_MATRIX_EQUAL(track_ids, frame_cloned.getTrackIds());

  EXPECT_NEAR_OPENCV(image, frame_cloned.getRawImage(), 0);

  EXPECT_TRUE(frame == frame_cloned);
}

TEST(Frame, getNormalizedBearingVectors) {
  // Create a test nframe with some keypoints.
  aslam::UnifiedProjectionCamera::Ptr camera = aslam::UnifiedProjectionCamera::createTestCamera();
  aslam::VisualFrame::Ptr frame = aslam::VisualFrame::createEmptyTestVisualFrame(camera, 0);

  const size_t kNumKeypoints = 10;
  Eigen::Matrix2Xd keypoints;
  keypoints.resize(Eigen::NoChange, kNumKeypoints);
  for (size_t idx = 0; idx < kNumKeypoints - 1; ++idx) {
    keypoints.col(idx) = camera->createRandomKeypoint();
  }
  keypoints.col(kNumKeypoints - 1) = Eigen::Vector2d(1e8, 1e8); // Add one invalid keypoint.
  frame->setKeypointMeasurements(keypoints);

  // Get bearing vectors.
  std::vector<size_t> keypoint_indices;
  keypoint_indices.emplace_back(1);
  keypoint_indices.emplace_back(3);
  keypoint_indices.emplace_back(2);
  keypoint_indices.emplace_back(4);
  keypoint_indices.emplace_back(kNumKeypoints - 1);  // This is the invalid keypoint.

  std::vector<unsigned char> projection_success;
  Eigen::Matrix3Xd bearing_vectors = frame->getNormalizedBearingVectors(keypoint_indices,
                                                                        &projection_success);

  // Check by manually calculating the normalized bearing vectors.
  const size_t num_bearing_vectors = static_cast<size_t>(bearing_vectors.cols());
  ASSERT_EQ(num_bearing_vectors, keypoint_indices.size());
  ASSERT_EQ(num_bearing_vectors, projection_success.size());

  for (size_t bearing_idx = 0; bearing_idx < num_bearing_vectors; ++bearing_idx) {
    // Manually backproject.
    Eigen::Vector3d point_3d;
    bool success_manual = camera->backProject3(keypoints.col(keypoint_indices[bearing_idx]),
                                               &point_3d);
    Eigen::Vector3d bearing_vector_manual = point_3d.normalized();

    Eigen::Vector3d bearing_vector = bearing_vectors.col(bearing_idx);
    EXPECT_TRUE(EIGEN_MATRIX_NEAR(bearing_vector, bearing_vector_manual, 1e-12));
    EXPECT_EQ(success_manual, projection_success[bearing_idx]);
  }
}

ASLAM_UNITTEST_ENTRYPOINT
