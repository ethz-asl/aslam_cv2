#ifndef ASLAM_GYRO_TRACKER_H_
#define ASLAM_GYRO_TRACKER_H_

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/macros.h>
#include <aslam/tracker/feature-tracker.h>


// Some forward declarations.
class BinaryFeatureStore;
namespace aslam { class VisualFrame; }
namespace aslam { class Camera; }

namespace aslam {
/// \class GyroTracker
/// \brief Feature tracker using an interframe rotation matrix to predict the feature positions
///        while matching.
///        /TODO(schneith): more details
class GyroTracker : public FeatureTracker {
 public:
  ASLAM_POINTER_TYPEDEFS(GyroTracker);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(GyroTracker);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  /// Constructor for serialization.
  GyroTracker() = default;

 public:
  /// \brief Construct the feature tracker.
  /// @param[in] input_camera The camera used in the tracker for projection/backprojection.
  GyroTracker(const std::shared_ptr<const aslam::Camera>& input_camera);
  virtual ~GyroTracker() {};

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
                        const Eigen::Matrix3d& C_current_prev);

 private:
  /// \brief Match features between the current and the pfvrevious frames using a given interframe
  ///        rotation C_current_prev to predict the feature positions.
  /// @param[in] C_current_prev Rotation matrix that describes the camera rotation between the
  ///                           two frames that are matched.
  /// @param[in] current_frame  The current VisualFrame that needs to contain the keypoints and
  ///                           descriptor channels. Usually this is an output of the VisualPipeline.
  /// @param[in] previous_frame The previous VisualFrame that needs to contain the keypoints and
  ///                           descriptor channels. Usually this is an output of the VisualPipeline.
  /// @param[out] matches_prev_current Vector of index pairs containing the found matches. Indices
  ///                           correspond to the ordering of the keypoint/descriptor vector in the
  ///                           respective frame channels. (pair.first = previous_frame, pair.second
  ///                           = current_frame)
  void matchFeatures(const Eigen::Matrix3d& C_current_prev,
                     const VisualFrame& current_frame,
                     const VisualFrame& previous_frame,
                     std::vector<std::pair<int, int> >* matches_prev_current) const;

  /// The frame processed in the last update.
  std::shared_ptr<const aslam::VisualFrame> previous_frame_ptr_;

  /// Track length corresponding to the feature vector in the current frame.
  std::vector<int> current_track_lengths_;
  /// Track length corresponding to the feature vector in the previous frame.
  std::vector<int> previous_track_lengths_;

  //TODO(schneith): add documentation for the parameters && make settings??
  const float kKeypointScoreThreshold = 5.0;
  const int kNumberOfTrackingBuckets = 4;
  const int kNumberOfKeyPointsUseUnconditional = 100;
  const float kKeypointScoreThresholdUnconditional = kKeypointScoreThreshold * 2;
  const int kNumberOfKeyPointsUseStrong = 1000;
  const float kKeypointScoreThresholdStrong = kKeypointScoreThreshold * 1.2;
};

}       // namespace aslam

#endif  // ASLAM_GYRO_TRACKER_H_
