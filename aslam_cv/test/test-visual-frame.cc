#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>

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
EXPECT_TRUE(EIGEN_MATRIX_NEAR(data, data_2, 1e-6));
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
