#ifndef ASLAM_VISUAL_MULTI_FRAME_H
#define ASLAM_VISUAL_MULTI_FRAME_H

#include <aslam/cameras/camera-rig.h>
#include <aslam/frames/visual-frame.h>

namespace aslam {

/// \class VisualMultiFrame
/// \brief A class representing images and keypoints and 
///        calibration data from a multi-camera system
class VisualMultiFrame {
 public:
  ASLAM_POINTER_TYPEDEFS(VisualMultiFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(VisualMultiFrame);

  /// \brief creates an empty visual multi frame
  VisualMultiFrame();
  
  virtual ~VisualMultiFrame();

  /// \brief get the multiframe id.
  inline aslam::MultiFrameId getId() const { return id_; }
  
  /// \brief set the multiframe id.
  inline void setId(aslam::MultiFrameId id) { id_ = id; }

  /// \brief get the timestamp
  inline uint64_t getTimestamp() const{ return stamp_; }
  
  /// \brief set the timestamp
  inline void setTimestamp(uint64_t stamp){ stamp_ = stamp; }

  /// \brief set the camera rig
  void setCameraRig(CameraRig::Ptr rig);

  /// \brief get the camera rig
  const CameraRig& getCameraRig() const;
  
  /// \brief get the camera rig
  CameraRig::Ptr getCameraRigMutable();
  
  /// \brief get one frame
  const VisualFrame& getFrame(size_t frameIndex) const;

  /// \brief get one frame
  VisualFrame::Ptr getFrameMutable(size_t frameIndex);

  /// \brief the number of frames
  size_t getNumFrames() const;

  /// \brief the number of frames
  size_t getNumCameras() const;

  /// \brief get the pose of body frame with respect to the camera i
  const Transformation& get_T_C_B(size_t cameraIndex) const;

  /// \brief get the geometry object for camera i
  const Camera& getCamera(size_t cameraIndex) const;

  /// \brief gets the id for the camera at index i
  CameraId getCameraId(size_t cameraIndex) const;
  
  /// \brief does this rig have a camera with this id
  bool hasCameraWithId(const CameraId& id) const;
  
  /// \brief get the index of the camera with the id
  /// @returns -1 if the rig doesn't have a camera with this id
  size_t getCameraIndex(const CameraId& id) const;

  /// \brief binary equality
  bool operator==(const VisualMultiFrame& other) const;

 private:
  /// integer nanoseconds since epoch
  uint64_t stamp_;
  
  /// \brief the cs frame id
  MultiFrameId id_;

  /// \brief the frame id
  CameraRig::Ptr cameraRig_;
  
  /// \brief the list of individual image frames
  std::vector<VisualFrame::Ptr> frames_;
};

} // namespace aslam


#endif /* ASLAM_VISUAL_MULTI_FRAME_H */
