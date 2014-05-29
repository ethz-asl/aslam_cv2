#include <aslam/frames/VisualFrame.h>

namespace aslam {
virtual bool VisualFrame::operator==(const VisualFrame& other) const {
  bool same = true;
  same &= keypoints_ == other.keypoints_;
  same &= descriptors_ == other.descriptors_;
  same &= static_cast<bool>(camera_geometry_) ==
      static_cast<bool>(other.camera_geometry_);
  if (same) {
    same &= *camera_geometry_ == *other.camera_geometry_;
  }
  return same;
}
const Eigen::Matrix2Xd& VisualFrame::getKeypoints() const {
  return ;
}
const VisualFrame::DescriptorsT& VisualFrame::getDescriptors() const {
  return descriptors_;
}
Eigen::Matrix2Xd* VisualFrame::getKeypointsMutable() {
  return &keypoints_;
}
DescriptorsT* VisualFrame::getDescriptorsMutable() {
  return &descriptors_;
}

const char* VisualFrame::getDescriptor(size_t index) const {
  CHECK_LT(index, descriptors_.cols());
  return &descriptors_.coeffRef(0, index);
}
const Eigen::Block<Eigen::Matrix2Xd, 2, 1> VisualFrame::getKeypoint(size_t index) const {
  CHECK_LT(index, keypoints_.cols());
  return keypoints_.block<2, 1>(0, index);
}

const void VisualFrame::setKeypoints(const Eigen::Matrix2Xd& keypoints) {
  keypoints_ = keypoints;
}
const void VisualFrame::setDescriptors(const DescriptorsT& descriptors) {
  descriptors_ = descriptors;
}

const std::shared_ptr<const cameras::Camera> VisualFrame::getCameraGeometry() const {
  return camera_geometry_;
}
void VisualFrame::setCameraGeometry(const std::shared_ptr<cameras::Camera>& camera) {
  camera_geometry_ = camera;
}

}  // namespace aslam
