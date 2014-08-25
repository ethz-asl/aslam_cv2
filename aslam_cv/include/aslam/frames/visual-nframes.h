#ifndef ASLAM_VISUAL_MULTI_FRAME_H
#define ASLAM_VISUAL_MULTI_FRAME_H

#include <aslam/cameras/ncameras.h>
#include <aslam/frames/visual-frame.h>

namespace aslam {

/// \class VisualMultiFrame
/// \brief A class representing images and keypoints and 
///        calibration data from a multi-camera system
class VisualNFrames {
 public:
  ASLAM_POINTER_TYPEDEFS(VisualNFrames);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(VisualNFrames);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief creates an empty visual multi frame
  VisualNFrames();
  
  virtual ~VisualNFrames();

  /// \brief get the multiframe id.
  inline const aslam::NFramesId& getId() const { return id_; }
  
  /// \brief set the multiframe id.
  inline void setId(const aslam::NFramesId& id) { id_ = id; }

  /// \brief set the camera rig
  void setNCameras(NCameras::Ptr rig);

  /// \brief get the camera rig
  const NCameras& getNCameras() const;
  
  /// \brief get the camera rig
  NCameras::Ptr getNCamerasMutable();
  
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
  const CameraId& getCameraId(size_t cameraIndex) const;
  
  /// \brief does this rig have a camera with this id
  bool hasCameraWithId(const CameraId& id) const;
  
  /// \brief get the index of the camera with the id
  /// @returns -1 if the rig doesn't have a camera with this id
  size_t getCameraIndex(const CameraId& id) const;

  /// \brief binary equality
  bool operator==(const VisualNFrames& other) const;

 private:
  /// \brief the cs frame id
  NFramesId id_;

  /// \brief the frame id
  NCameras::Ptr cameraRig_;
  
  /// \brief the list of individual image frames
  std::vector<VisualFrame::Ptr> frames_;
};

} // namespace aslam


#endif /* ASLAM_VISUAL_MULTI_FRAME_H */
