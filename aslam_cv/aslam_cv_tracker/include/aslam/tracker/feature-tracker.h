#ifndef ASLAM_FEATURE_TRACKER_BASE_H_
#define ASLAM_FEATURE_TRACKER_BASE_H_

#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>
#include <aslam/matcher/match.h>
#include <Eigen/Dense>
#include <glog/logging.h>

namespace aslam {
class VisualFrame;
}

namespace aslam {
/// \class FeatureTracker
/// \brief Base class defining the interface for feature trackers and providing visualization.
class FeatureTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ASLAM_POINTER_TYPEDEFS(FeatureTracker);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(FeatureTracker);

 protected:
  FeatureTracker() = default;
 public:
  virtual ~FeatureTracker() {};

  /// Track features and return the matches. The matches are not written to the trackid channels
  /// and should be written to the track id channels using a TrackManager (after e.g. outlier
  /// filtering).
  virtual void track(const std::shared_ptr<aslam::VisualFrame>& frame_kp1,
                     const std::shared_ptr<aslam::VisualFrame>& frame_k,
                     const aslam::Quaternion& q_Ckp1_Ck,
                     aslam::MatchesWithScore* matches_with_score_kp1_k) = 0;
};

}  // namespace aslam

#endif  // ASLAM_FEATURE_TRACKER_BASE_H_
