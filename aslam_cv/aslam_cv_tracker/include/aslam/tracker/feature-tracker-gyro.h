#ifndef ASLAM_GYRO_TRACKER_H_
#define ASLAM_GYRO_TRACKER_H_

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/macros.h>
#include <aslam/tracker/feature-tracker.h>

namespace aslam {
class VisualFrame;
class Camera;
}

namespace aslam {
/// \class GyroTracker
/// \brief Feature tracker using an interframe rotation matrix to predict the feature positions
///        while matching.
class GyroTracker : public FeatureTracker{
 public:
  ASLAM_POINTER_TYPEDEFS(GyroTracker);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(GyroTracker);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  /// \brief Construct the feature tracker.
  /// @param[in] input_camera The camera used in the tracker for projection/backprojection.
  explicit GyroTracker(const aslam::Camera& camera);
  virtual ~GyroTracker() {}

  virtual void track(const aslam::Quaternion& q_Ckp1_Ck,
                     const aslam::VisualFrame& frame_k,
                     aslam::VisualFrame* frame_kp1,
                     aslam::MatchesWithScore* matches_with_score_kp1_k) override;

 private:
  /// \brief Match features between the current and the previous frames using a given interframe
  ///        rotation q_Ckp1_Ck to predict the feature positions.
  /// @param[in] q_Ckp1_Ck      Rotation matrix that describes the camera rotation between the
  ///                           two frames that are matched.
  /// @param[in] frame_kp1      The current VisualFrame that needs to contain the keypoints and
  ///                           descriptor channels. Usually this is an output of the VisualPipeline.
  /// @param[in] frame_k        The previous VisualFrame that needs to contain the keypoints and
  ///                           descriptor channels. Usually this is an output of the VisualPipeline.
  /// @param[out] matches_prev_current  Vector of structs containing the found matches. Indices
  ///                                   correspond to the ordering of the keypoint/descriptor vector in the
  ///                                   respective frame channels. (Apple = frame_kp1, Banana = frame_k)
  void matchFeatures(const aslam::Quaternion& q_Ckp1_Ck,
                     const VisualFrame& frame_k,
                     const VisualFrame& frame_kp1,
                     aslam::MatchesWithScore* matches_with_score_kp1_k) const;


  /// The camera model used in the tracker.
  const aslam::Camera& camera_;

  /// Two descriptors can match if the number of matching bits normalized
  /// with the descriptor length in bits is higher than this threshold.
  const float kMatchingThresholdBitsRatio = 0.8;
};

}       // namespace aslam

#endif  // ASLAM_GYRO_TRACKER_H_
