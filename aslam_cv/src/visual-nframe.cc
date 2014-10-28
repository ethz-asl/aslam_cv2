#include <aslam/frames/visual-nframe.h>
#include <aslam/common/predicates.h>
namespace aslam {

VisualNFrame::VisualNFrame() { }

VisualNFrame::VisualNFrame(const aslam::NFramesId& id,
                           std::shared_ptr<NCamera> ncameras)
: id_(id), cameraRig_(ncameras) {
  CHECK_NOTNULL(cameraRig_.get());
  size_t num_frames = ncameras->getNumCameras();
  frames_.resize(num_frames);
  for (size_t c = 0; c < num_frames; ++c) {
    frames_[c] = std::make_shared<VisualFrame>();
  }
}

VisualNFrame::VisualNFrame(std::shared_ptr<NCamera> ncameras)
: cameraRig_(ncameras) {
  CHECK_NOTNULL(cameraRig_.get());
  id_.randomize();
  size_t num_frames = ncameras->getNumCameras();
  frames_.resize(num_frames);
  for (size_t c = 0; c < num_frames; ++c) {
    frames_[c] = std::make_shared<VisualFrame>();
  }
}

VisualNFrame::~VisualNFrame() { }

/// \brief get the camera rig
const NCamera& VisualNFrame::getNCameras() const {
  CHECK_NOTNULL(cameraRig_.get());
  return *cameraRig_;
}

/// \brief get the camera rig
NCamera::Ptr VisualNFrame::getNCamerasMutable() {
  return cameraRig_;
}

/// \brief get one frame
const VisualFrame& VisualNFrame::getFrame(size_t frameIndex) const {
  CHECK_LT(frameIndex, frames_.size());
  CHECK_NOTNULL(frames_[frameIndex].get());
  return *frames_[frameIndex];
}

/// \brief get one frame
VisualFrame::Ptr VisualNFrame::getFrameMutable(size_t frameIndex) {
  CHECK_LT(frameIndex, frames_.size());
  return frames_[frameIndex];  
}

/// \brief the number of frames
size_t VisualNFrame::getNumFrames() const {
  return frames_.size();
}

/// \brief the number of frames
size_t VisualNFrame::getNumCameras() const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getNumCameras();
}

/// \brief get the pose of body frame with respect to the camera i
const Transformation& VisualNFrame::get_T_C_B(size_t cameraIndex) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->get_T_C_B(cameraIndex);
}

/// \brief get the geometry object for camera i
const Camera& VisualNFrame::getCamera(size_t cameraIndex) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getCamera(cameraIndex);
}

/// \brief gets the id for the camera at index i
const CameraId& VisualNFrame::getCameraId(size_t cameraIndex) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getCameraId(cameraIndex);
}

/// \brief does this rig have a camera with this id
bool VisualNFrame::hasCameraWithId(const CameraId& id) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->hasCameraWithId(id);
}

/// \brief get the index of the camera with the id
/// @returns -1 if the rig doesn't have a camera with this id
size_t VisualNFrame::getCameraIndex(const CameraId& id) const {
  CHECK_NOTNULL(cameraRig_.get());
  return cameraRig_->getCameraIndex(id);
}

void VisualNFrame::setFrame(size_t frameIndex, VisualFrame::Ptr frame) {
  CHECK_LT(frameIndex, frames_.size());
  CHECK_NOTNULL(cameraRig_.get());
  CHECK_EQ(&cameraRig_->getCamera(frameIndex), frame->getCameraGeometry().get());
  frames_[frameIndex] = frame;
}

bool VisualNFrame::isFrameNull(size_t frameIndex) const {
  CHECK_LT(frameIndex, frames_.size());
  return static_cast<bool>(frames_[frameIndex]);
}

/// \brief binary equality
bool VisualNFrame::operator==(const VisualNFrame& other) const {
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
