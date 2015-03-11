#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/feature-tracker-descriptor-matcher.h>

namespace aslam {

void FeatureTrackerDescriptorMatching::track(const std::shared_ptr<aslam::VisualFrame>& frame_kp1,
                                             const std::shared_ptr<aslam::VisualFrame>& frame_k,
                                             const aslam::Quaternion& q_Ckp1_Ck,
                                             aslam::MatchesWithScore* matches_with_score_kp1_k) {
  CHECK(frame_k);
  CHECK(frame_kp1);
  CHECK_NOTNULL(matches_with_score_kp1_k);
  CHECK(frame_kp1->hasDescriptors());
  CHECK(frame_kp1->hasKeypointMeasurements());
  CHECK(frame_k->hasDescriptors());
  CHECK(frame_k->hasKeypointMeasurements());

  aslam::MatchingProblemFrameToFrame matching_problem(frame_kp1, frame_k, q_Ckp1_Ck,
                                                      kImageSpaceDistanceThreshold,
                                                      kDescriptorDistanceThreshold);

  // Match keypoints to previous frame of this camera.
  matching_engines_.match(&matching_problem, matches_with_score_kp1_k);
}

}  // namespace aslam

