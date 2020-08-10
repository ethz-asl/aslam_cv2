#include <aslam/common/channel-definitions-lidar.h>
#include <aslam/common/stl-helpers.h>
#include <aslam/common/time.h>
#include <memory>

#include "aslam/frames/visual-frame.h"

namespace aslam {

bool VisualFrame::hasLidarKeypointPositions() const {
  return aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_);
}

const Eigen::Matrix3Xd& VisualFrame::getKeypointVectors() const {
  return aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
}

Eigen::Matrix3Xd* VisualFrame::getKeypointVectorsMutable() {
  Eigen::Matrix3Xd& vector =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  return &vector;
}

const Eigen::Block<Eigen::Matrix3Xd, 3, 1> VisualFrame::getKeypointVector(
    size_t index) const {
  Eigen::Matrix3Xd& keypoints =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  CHECK_LT(static_cast<int>(index), keypoints.cols());
  return keypoints.block<3, 1>(0, index);
}

void VisualFrame::setKeypointVectors(const Eigen::Matrix3Xd& vectors_new) {
  if (!aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_)) {
    aslam::channels::add_KEYPOINT_VECTORS_Channel(&channels_);
  }
  Eigen::Matrix3Xd& vectors =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  vectors = vectors_new;
}

void VisualFrame::swapKeypointVectors(Eigen::Matrix3Xd* vectors_new) {
  if (!aslam::channels::has_KEYPOINT_VECTORS_Channel(channels_)) {
    aslam::channels::add_KEYPOINT_VECTORS_Channel(&channels_);
  }
  Eigen::Matrix3Xd& vectors =
      aslam::channels::get_KEYPOINT_VECTORS_Data(channels_);
  vectors.swap(*vectors_new);
}
}  // namespace aslam
