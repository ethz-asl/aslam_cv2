#ifndef ASLAM_FEATURE_TRACKER_DESCRIPTOR_MATCHING_H_
#define ASLAM_FEATURE_TRACKER_DESCRIPTOR_MATCHING_H_

#include <vector>

#include <aslam/common/memory.h>
#include <aslam/matcher/match.h>
#include <aslam/matcher/matching-engine-exclusive.h>
#include <aslam/matcher/matching-problem-frame-to-frame.h>
#include <aslam/tracker/feature-tracker.h>
#include <Eigen/Dense>

namespace aslam {
class VisualFrame;

class FeatureTrackerDescriptorMatching : public FeatureTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef aslam::Aligned<std::vector, Eigen::Vector2d>::type Vector2dList;

  FeatureTrackerDescriptorMatching() = default;
  virtual ~FeatureTrackerDescriptorMatching() {
  }

  /// \brief Tracker that matches brisk descriptors in a small search window that is predicted
  ///        using the given rotation.
  /// @param[in] q_Ckp1_Ck Rotation taking points from the Ck frame to the Ckp1 frame.
  /// @param[in] frame_k Frame at time step k.
  /// @param[in,out] frame_kp1 Frame at time step k+1.
  /// @param[out] matches_with_score_kp1_k Detected matches from frame k to frame k+1.
  virtual void track(
      const Quaternion& q_Ckp1_Ck, const VisualFrame& frame_k, VisualFrame* frame_kp1,
      MatchingProblemFrameToFrame::MatchesWithScore* matches_with_score_kp1_k) override;

 private:
  MatchingEngineExclusive<MatchingProblemFrameToFrame> matching_engines_;

  /// Max. image space distance for keypoint matches.
  static constexpr double kImageSpaceDistanceThreshold = 50.0;
  /// Max. descriptor distance for keypoint matches.
  static constexpr int kDescriptorDistanceThreshold = 120;
};
}  // namespace aslam

#endif  // ASLAM_FEATURE_TRACKER_DESCRIPTOR_MATCHING_H_
