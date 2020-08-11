#include <aslam/common/channel-definitions-lidar.h>
#include <aslam/common/stl-helpers.h>
#include <aslam/common/time.h>
#include <memory>

#include "aslam/frames/visual-frame.h"

namespace aslam {

bool VisualFrame::hasLidarKeypoint3DMeasurements() const {
  return aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_);
}

bool VisualFrame::hasLidarKeypoint2DMeasurements() const {
  return aslam::channels::has_LIDAR_2D_MEASUREMENTS_Channel(channels_);
}

bool VisualFrame::hasLidarDescriptors() const {
  return aslam::channels::has_LIDAR_DESCRIPTORS_Channel(channels_);
}

const Eigen::Matrix3Xd& VisualFrame::getLidarKeypoint3DMeasurements() const {
  return aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
}

const Eigen::Matrix2Xd& VisualFrame::getLidarKeypoint2DMeasurements() const {
  return aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
}

const VisualFrame::DescriptorsT& VisualFrame::getLidarDescriptors() const {
  return aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
}

Eigen::Matrix3Xd* VisualFrame::getLidarKeypoint3DMeasurementsMutable() {
  Eigen::Matrix3Xd& vector =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  return &vector;
}

Eigen::Matrix2Xd* VisualFrame::getLidarKeypoint2DMeasurementsMutable() {
  Eigen::Matrix2Xd& vector =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  return &vector;
}

VisualFrame::DescriptorsT* VisualFrame::getLidarDescriptorsMutable() {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  return &descriptors;
}

const Eigen::Block<Eigen::Matrix3Xd, 3, 1>
VisualFrame::getLidarKeypoint3DMeasurement(size_t index) const {
  Eigen::Matrix3Xd& keypoints =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  CHECK_LT(static_cast<int>(index), keypoints.cols());
  return keypoints.block<3, 1>(0, index);
}

const Eigen::Block<Eigen::Matrix2Xd, 2, 1>
VisualFrame::getLidarKeypoint2DMeasurement(size_t index) const {
  Eigen::Matrix2Xd& keypoints =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  CHECK_LT(static_cast<int>(index), keypoints.cols());
  return keypoints.block<2, 1>(0, index);
}

const unsigned char* VisualFrame::getLidarDescriptor(size_t index) const {
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  CHECK_LT(static_cast<int>(index), descriptors.cols());
  return &descriptors.coeffRef(0, index);
}

void VisualFrame::setLidarKeypoint3DMeasurements(
    const Eigen::Matrix3Xd& vectors_new) {
  if (!aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_)) {
    aslam::channels::add_KEYPOINT_VECTORS_Channel(&channels_);
  }
  Eigen::Matrix3Xd& vectors =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  vectors = vectors_new;
}

void VisualFrame::setLidarKeypoint2DMeasurements(
    const Eigen::Matrix2Xd& vectors_new) {
  if (!aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_)) {
    aslam::channels::add_LIDAR_2D_MEASUREMENTS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& vectors =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  vectors = vectors_new;
}

void VisualFrame::setLidarDescriptors(
    const Eigen::Map<const DescriptorsT>& descriptors_new) {
  if (!aslam::channels::has_LIDAR_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_LIDAR_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  descriptors = descriptors_new;
}

void VisualFrame::swapLidarKeypoint3DMeasurements(
    Eigen::Matrix3Xd* vectors_new) {
  if (!aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_)) {
    aslam::channels::add_KEYPOINT_VECTORS_Channel(&channels_);
  }
  Eigen::Matrix3Xd& vectors =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  vectors.swap(*vectors_new);
}

void VisualFrame::swapLidarKeypoint2DMeasurements(
    Eigen::Matrix2Xd* vectors_new) {
  if (!aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_)) {
    aslam::channels::add_KEYPOINT_VECTORS_Channel(&channels_);
  }
  Eigen::Matrix2Xd& vectors =
      aslam::channels::get_LIDAR_2D_MEASUREMENTS_Data(channels_);
  vectors.swap(*vectors_new);
}

void VisualFrame::swapLidarDescriptors(DescriptorsT* descriptors_new) {
  if (!aslam::channels::has_LIDAR_DESCRIPTORS_Channel(channels_)) {
    aslam::channels::add_LIDAR_DESCRIPTORS_Channel(&channels_);
  }
  VisualFrame::DescriptorsT& descriptors =
      aslam::channels::get_LIDAR_DESCRIPTORS_Data(channels_);
  descriptors.swap(*descriptors_new);
}
}  // namespace aslam
