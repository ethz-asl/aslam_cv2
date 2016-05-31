#include "aslam/tracker/feature-tracker-gyro.h"

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/common/memory.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/gyro-two-frame-matcher.h>

#include "aslam/tracker/tracking-helpers.h"

namespace aslam {

GyroTracker::GyroTracker(const Camera& camera)
    : camera_(camera) {}

void GyroTracker::track(const Quaternion& q_Ckp1_Ck,
                        const VisualFrame& frame_k,
                        VisualFrame* frame_kp1,
                        MatchesWithScore* matches_with_score_kp1_k) {
  CHECK(frame_k.hasKeypointMeasurements());
  CHECK(CHECK_NOTNULL(frame_kp1)->hasKeypointMeasurements());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(frame_k.getCameraGeometry().get())->getId());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(frame_kp1->getCameraGeometry().get())->getId());
  CHECK_NOTNULL(matches_with_score_kp1_k)->clear();
  CHECK(frame_k.hasTrackIds());
  CHECK(frame_kp1->hasTrackIds());
  // Make sure the frames are in order time-wise
  CHECK_GT(frame_kp1->getTimestampNanoseconds(), frame_k.getTimestampNanoseconds());
  // Check that the required data is available in the frame
  CHECK(frame_kp1->hasDescriptors());
  CHECK(frame_k.hasDescriptors());
  CHECK_EQ(frame_kp1->getDescriptors().rows(), frame_kp1->getDescriptorSizeBytes());
  CHECK_EQ(frame_kp1->getKeypointMeasurements().cols(), frame_kp1->getDescriptors().cols());

  // Match the descriptors of frame (k+1) with those of frame k.
  GyroTwoFrameMatcher matcher(
      q_Ckp1_Ck, *frame_kp1, frame_k,
      camera_.imageHeight(), matches_with_score_kp1_k);
  matcher.Match();
}

}  //namespace aslam
