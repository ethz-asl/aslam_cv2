#ifndef ASLAM_FEATURE_TRACKER_BASE_H_
#define ASLAM_FEATURE_TRACKER_BASE_H_

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/common/macros.h>

namespace aslam { class VisualFrame; }

namespace aslam {
/// \class FeatureTracker
/// \brief Base class defining the interface for feature trackers and providing visualization.
class FeatureTracker {
 public:
  ASLAM_POINTER_TYPEDEFS(FeatureTracker);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(FeatureTracker);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  /// Constructor for serialization.
  FeatureTracker();

 protected:
  FeatureTracker(const aslam::Camera::ConstPtr& input_camera);

 public:
  virtual ~FeatureTracker() {};

  struct FeatureMatch {
    explicit FeatureMatch(int index_previous_frame, int index_current_frame)
    : index_previous_frame_(index_previous_frame),
      index_current_frame_(index_current_frame) {}
    int index_previous_frame_;
    int index_current_frame_;
  };

  struct MatchCandidateScore {
    explicit MatchCandidateScore(int index, double score)
    : match_index_(index),
      score_(score) {}
    int match_index_;
    double score_;
  };

  /// \brief Add a new VisualFrame to the tracker.
  ///        The frame needs to contain KeypointMeasurements and Descriptors channel. Usually
  ///        this VisualFrame will be an output of a VisualPipeline. The tracker will fill the
  ///        TrackId channel with the tracking results.
  ///        NOTE: Make sure the frames arrive in correct time ordering.
  /// @param[in] current_frame_ptr Shared pointer to the frame to process. The frame needs to
  ///                              contain KeypointMeasurements and Descriptors channel. Usually
  ///                              this VisualFrame will be an output of a VisualPipeline.
  /// @param[in] C_current_prev    Rotation matrix that defines the camera rotation between the
  ///                              current and the previous frame passed to the tracker. For e.g.
  ///                              this could be the output of the gyro/odometry integrators.
  virtual void addFrame(std::shared_ptr<VisualFrame> current_frame_ptr,
                        const Eigen::Matrix3d& C_current_prev) = 0;

  /// \brief Add a new VisualFrame to the tracker and draw tracks to an image.
  ///        The frame needs to contain KeypointMeasurements and Descriptors channel. Usually
  ///        this VisualFrame will be an output of a VisualPipeline. The tracker will fill the
  ///        TrackId channel with the tracking results.
  ///        NOTE: Make sure the frames arrive in correct time ordering.
  /// @param[in] current_frame_ptr Shared pointer to the frame to process. The frame needs to
  ///                              contain KeypointMeasurements and Descriptors channel. Usually
  ///                              this VisualFrame will be an output of a VisualPipeline.
  /// @param[in] C_current_prev    Rotation matrix that defines the camera rotation between the
  ///                              current and the previous frame passed to the tracker. For e.g.
  ///                              this could be the output of the gyro/odometry integrators.
  /// @param[out] track_image      Tracks are drawn into this image.
  void addFrameAndDrawTracks(std::shared_ptr<VisualFrame> current_frame_ptr,
                             const Eigen::Matrix3d& C_current_prev,
                             cv::Mat* track_image) {
    addFrame(current_frame_ptr, C_current_prev);
    drawTracks(*current_frame_ptr.get(), track_image);
  }

  /// \brief Draw feature tracks to an image.
  /// @param[in] current_frame  The frame to process. The frame needs to contain a TrackId channel.
  /// @param[out] track_image   Tracks are drawn into this image.
  void drawTracks(const VisualFrame& current_frame, cv::Mat* track_image);

  /// Return the used camera model.
  const aslam::Camera& getCamera() const { return *CHECK_NOTNULL(camera_.get()); };

 protected:
  /// The camera model used in the tracker.
  std::shared_ptr<const aslam::Camera> camera_;

  /// Track id provider
  unsigned int current_track_id_;

 private:
  /// Track type for drawing.
  struct Track {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Track() : track_id(-1), length(0),
              starting_point(Eigen::Vector2d::Zero()),
              end_point(Eigen::Vector2d::Zero()) {};
    int track_id;
    int length;
    Eigen::Vector2d starting_point;
    Eigen::Vector2d end_point;
  };

  /// The track ids drawn in the last update.
  std::map<int, Track> previous_drawn_tracks_;
};

} // namespace aslam

#endif  // ASLAM_FEATURE_TRACKER_BASE_H_
