#include <aslam/frames/visual-nframes.h>
#include <aslam/common/predicates.h>
namespace aslam {

VisualNFrames::VisualNFrames() { }

VisualNFrames::VisualNFrames(const aslam::NFramesId& id,
                             std::shared_ptr<NCameras> ncameras) :
                                   id_(id), cameraRig_(ncameras) {
  CHECK_NOTNULL(cameraRig_.get());
  frames_.resize(ncameras->getNumCameras());
}

VisualNFrames::VisualNFrames(std::shared_ptr<NCameras> ncameras) :
  cameraRig_(ncameras) {
  CHECK_NOTNULL(cameraRig_.get());
  id_.randomize();
  frames_.resize(ncameras->getNumCameras());
}
  
VisualNFrames::~VisualNFrames() { }

/// \brief get the camera rig
const NCameras& VisualNFrames::getNCameras() const {
  CHECK_NOTNULL(cameraRig_.get());
  return *cameraRig_;
}
  
/// \brief get the camera rig
NCameras::Ptr VisualNFrames::getNCamerasMutable() {
  return cameraRig_;
}
  
/// \brief get one frame
const VisualFrame& VisualNFrames::getFrame(size_t frameIndex) const {
  CHECK_LT(frameIndex, frames_.size());
  CHECK_NOTNULL(frames_[frameIndex].get());
  return *frames_[frameIndex];
}

/// \brief get one frame
VisualFrame::Ptr VisualNFrames::getFrameMutable(size_t frameIndex) {
  CHECK_LT(frameIndex, frames_.size());
  return frames_[frameIndex];  
}

/// \brief the number of frames
size_t VisualNFrames::getNumFrames() const {
  return frames_.size();
}

/// \brief the number of frames
size_t VisualNFrames::getNumCameras() const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getNumCameras();
}

/// \brief get the pose of body frame with respect to the camera i
const Transformation& VisualNFrames::get_T_C_B(size_t cameraIndex) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->get_T_C_B(cameraIndex);
}

/// \brief get the geometry object for camera i
const Camera& VisualNFrames::getCamera(size_t cameraIndex) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getCamera(cameraIndex);
}

/// \brief gets the id for the camera at index i
const CameraId& VisualNFrames::getCameraId(size_t cameraIndex) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getCameraId(cameraIndex);
}
  
/// \brief does this rig have a camera with this id
bool VisualNFrames::hasCameraWithId(const CameraId& id) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->hasCameraWithId(id);
}
  
/// \brief get the index of the camera with the id
/// @returns -1 if the rig doesn't have a camera with this id
size_t VisualNFrames::getCameraIndex(const CameraId& id) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getCameraIndex(id);
}

void VisualNFrames::setFrame(size_t frameIndex, VisualFrame::Ptr frame) {
  CHECK_LT(frameIndex, frames_.size());
  CHECK_NOTNULL(cameraRig_.get());
  CHECK_EQ(&cameraRig_->getCamera(frameIndex), frame->getCameraGeometry().get());
  frames_[frameIndex] = frame;
}

bool VisualNFrames::isFrameNull(size_t frameIndex) const {
  CHECK_LT(frameIndex, frames_.size());
  return static_cast<bool>(frames_[frameIndex]);
}

/// \brief binary equality
bool VisualNFrames::operator==(const VisualNFrames& other) const {
  bool same = true;
  
  same &= id_ == other.id_;
  same &= aslam::checkSharedEqual(cameraRig_, other.cameraRig_);
  same &= frames_.size() == other.frames_.size();
  if(same) {
    for(size_t i = 0; i < frames_.size(); ++i) {
      same &= aslam::checkSharedEqual(frames_[i], other.frames_[i]);
    }
  }
  return same;
}


} // namespace aslam
