#ifndef ASLAM_VISUAL_MULTI_FRAME_H
#define ASLAM_VISUAL_MULTI_FRAME_H

#include <aslam/cameras/ncameras.h>
#include <aslam/frames/visual-frame.h>

namespace aslam {

/// \class VisualMultiFrame
/// \brief A class representing images and keypoints and 
///        calibration data from a multi-camera system.
class VisualNFrames {
 public:
  ASLAM_POINTER_TYPEDEFS(VisualNFrames);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(VisualNFrames);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Creates an empty visual multi frame.
  VisualNFrames();
  
  /// \brief Creates a visual n-frame from an id and camera system.
  ///
  ///        The individual frames are initialized to NULL.
  ///
  /// \param[in] id       The unique id for this object.
  /// \param[in] ncameras The camera system associated with this object.
  VisualNFrames(const aslam::NFramesId& id,
                std::shared_ptr<NCameras> ncameras);

  /// \brief Creates a visual n-frame from a camera system.
  ///
  ///        The id is set randomly and the individual frames are initialized to NULL.
  ///
  /// \param[in] ncameras The camera system associated with this object.
  VisualNFrames(std::shared_ptr<NCameras> ncameras);

  virtual ~VisualNFrames();

  /// \brief Get the multiframe id.
  inline const aslam::NFramesId& getId() const { return id_; }

  /// \brief Get the camera rig.
  const NCameras& getNCameras() const;
  
  /// \brief Get the camera rig.
  NCameras::Ptr getNCamerasMutable();
  
  /// \brief Is the frame at this index null
  bool isFrameNull(size_t frameIndex) const;

  /// \brief Get one frame.
  const VisualFrame& getFrame(size_t frameIndex) const;

  /// \brief Get one frame, mutable.
  VisualFrame::Ptr getFrameMutable(size_t frameIndex);

  /// \brief Set the frame at the index.
  ///
  /// The method will fail hard if the frame does not have the same camera
  /// as specified in the camera system. It is expected that this method will
  /// mostly be used by the pipeline code when building a VisualNFrame for the
  /// first time.
  void setFrame(size_t frameIndex, VisualFrame::Ptr frame);

  /// \brief The number of frames.
  size_t getNumFrames() const;

  /// \brief The number of frames.
  size_t getNumCameras() const;

  /// \brief Get the pose of body frame with respect to the camera i.
  const Transformation& get_T_C_B(size_t cameraIndex) const;

  /// \brief Get the geometry object for camera i.
  const Camera& getCamera(size_t cameraIndex) const;

  /// \brief Get the id for the camera at index i.
  const CameraId& getCameraId(size_t cameraIndex) const;
  
  /// \brief Does this rig have a camera with this id?
  bool hasCameraWithId(const CameraId& id) const;
  
  /// \brief Get the index of the camera with the id.
  /// @returns -1 if the rig doesn't have a camera with this id
  size_t getCameraIndex(const CameraId& id) const;

  /// \brief Binary equality.
  bool operator==(const VisualNFrames& other) const;

 private:
  /// \brief The unique frame id.
  NFramesId id_;

  /// \brief The camera rig.
  NCameras::Ptr cameraRig_;
  
  /// \brief The list of individual image frames.
  std::vector<VisualFrame::Ptr> frames_;
};
} // namespace aslam

#endif /* ASLAM_VISUAL_MULTI_FRAME_H */
