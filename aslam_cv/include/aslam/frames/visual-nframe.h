#ifndef ASLAM_VISUAL_MULTI_FRAME_H
#define ASLAM_VISUAL_MULTI_FRAME_H

#include <aslam/cameras/ncamera.h>
#include <aslam/frames/visual-frame.h>

namespace aslam {

/// \class VisualMultiFrame
/// \brief A class representing images and keypoints and 
///        calibration data from a multi-camera system.
class VisualNFrame {
 public:
  ASLAM_POINTER_TYPEDEFS(VisualNFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(VisualNFrame);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::vector<VisualNFrame::Ptr> PtrVector;

 protected:
  /// \brief Creates an empty visual multi frame.
  VisualNFrame();

 public:
  /// \brief Creates a visual n-frame from an id and number of frames.
  ///
  /// This constructor should only be used in a specific situation when there
  /// is no information about the camera system when constructing the object.
  /// This may happen e.g. when deserializing data.
  ///
  ///        The individual frames are initialized to NULL.
  ///        The camera system is initialized to NULL.
  ///
  /// \param[in] id          The unique id for this object.
  /// \param[in] num_frames  The number of frames to be constructed.
  VisualNFrame(const aslam::NFramesId& id, unsigned int num_frames);

  /// \brief Creates a visual n-frame from an id and camera system.
  ///
  ///        The individual frames are initialized to NULL.
  ///
  /// \param[in] id       The unique id for this object.
  /// \param[in] ncameras The camera system associated with this object.
  VisualNFrame(const aslam::NFramesId& id,
                std::shared_ptr<NCamera> ncameras);

  /// \brief Creates a visual n-frame from a camera system.
  ///
  ///        The id is set randomly and the individual frames are initialized to NULL.
  ///
  /// \param[in] ncameras The camera system associated with this object.
  VisualNFrame(std::shared_ptr<NCamera> ncameras);

  virtual ~VisualNFrame();

  /// \brief Get the multiframe id.
  inline const aslam::NFramesId& getId() const { return id_; }

  /// \brief Set the multiframe id.
  inline void setId(const aslam::NFramesId& n_frames_id) { id_ = n_frames_id; }

  /// \brief Get the camera rig.
  const NCamera& getNCameras() const;
  
  /// \brief Get the camera rig.
  NCamera::Ptr getNCamerasMutable();
  
  /// \brief Set the camera rig.
  ///
  /// This method fills in in multi-camera system information. It should be
  /// used if we had no such knowledge at time of construction of this object.
  /// This method will also assign cameras to the already existing visual
  /// frames.
  void setNCameras(NCamera::Ptr ncameras);

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

  /// \brief Get the min timestamp in nanoseconds over all frames.
  int64_t getMinTimestampNanoseconds() const;

  /// \brief Binary equality.
  bool operator==(const VisualNFrame& other) const;

 private:
  /// \brief The unique frame id.
  NFramesId id_;

  /// \brief The camera rig.
  NCamera::Ptr cameraRig_;
  
  /// \brief The list of individual image frames.
  std::vector<VisualFrame::Ptr> frames_;
};
} // namespace aslam

#endif /* ASLAM_VISUAL_MULTI_FRAME_H */
