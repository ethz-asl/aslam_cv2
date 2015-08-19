#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/feature-tracker-lk.h>
#include <aslam/tracker/tracking-helpers.h>
#include <brisk/brisk.h>
#include <gflags/gflags.h>
#include <opencv/highgui.h>

DEFINE_bool(lk_show_detection_mask, false, "Draw the detection mask.");

namespace aslam {

FeatureTrackerLk::FeatureTrackerLk(const aslam::Camera& camera)
    : camera_(camera) {
  initialize(camera);
}

void FeatureTrackerLk::initialize(const aslam::Camera& camera) {
  // Create a detection mask that prevents detecting new keypoints close to the image border as
  // no descriptors can be calculated in this region.
  CHECK_LT(2 * kMinDistanceToImageBorderPx, camera.imageWidth());
  CHECK_LT(2 * kMinDistanceToImageBorderPx, camera.imageHeight());
  detection_mask_image_border_ = cv::Mat::zeros(camera.imageHeight(), camera.imageWidth(), CV_8UC1);
  cv::Mat region_of_interest(detection_mask_image_border_,
                             cv::Rect(kMinDistanceToImageBorderPx + 1,
                                      kMinDistanceToImageBorderPx + 1,
                                      camera.imageWidth() - 2 * kMinDistanceToImageBorderPx - 1,
                                      camera.imageHeight() - 2 * kMinDistanceToImageBorderPx - 1));
  region_of_interest = cv::Scalar(255);
}

void FeatureTrackerLk::track(const aslam::Quaternion& q_Ckp1_Ck,
                             const aslam::VisualFrame& frame_k,
                             aslam::VisualFrame* frame_kp1,
                             aslam::MatchesWithScore* matches_with_score_kp1_k) {
  aslam::timing::Timer timer_tracking("FeatureTrackerLk: track");
  CHECK_NOTNULL(frame_kp1);
  CHECK_NOTNULL(matches_with_score_kp1_k)->clear();
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(frame_k.getCameraGeometry().get())->getId());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(frame_kp1->getCameraGeometry().get())->getId());

  // Make sure the frame_kp1 does not yet contain keypoint/tracking information.
  CHECK(frame_kp1->hasRawImage());
  CHECK(!frame_kp1->hasKeypointMeasurements());
  CHECK(!frame_kp1->hasTrackIds());

  // Make sure the externally set list of keypoints to abort corresponds to this frame.
  if (!keypoint_indices_to_abort_.empty() && abort_keypoints_wrt_frame_id_ != frame_k.getId()) {
    LOG(FATAL) << "Keypoints to abort do not match the processed frame.";
  }

  // Track existing keypoints from frame (k) to frame (k+1).
  Vector2dList tracked_keypoints_kp1;
  std::vector<unsigned char> tracking_success;
  std::vector<float> tracking_errors;
  trackKeypoints(q_Ckp1_Ck, frame_k, frame_kp1->getRawImage(), &tracked_keypoints_kp1,
                 &tracking_success, &tracking_errors);

  // Reject tracked keypoints that meet one of the following criteria:
  //   - tracking was unsuccessful
  //   - marked for abortion by the external 2pt-ransac
  //   - too close to the image border
  //   - too close to other tracked point. In case of conflicts the keypoint
  //     with the lowest tracking error will be kept.
  // Reject features that got too close to each other (probably by object occlusion), left the
  // image, tracking failed or were marked for abortion by external logic. Keep the features with
  // the lower tracking error in case of conflicts.
  aslam::timing::Timer timer_selection("FeatureTrackerLk: track - feature selection");

  OccupancyGrid occupancy_grid(camera_.imageHeight(),camera_.imageWidth(),
                               kMinDistanceBetweenKeypointsPx, kMinDistanceBetweenKeypointsPx);

  for (size_t keypoint_idx_k = 0u; keypoint_idx_k < tracked_keypoints_kp1.size(); ++keypoint_idx_k) {
    // Drop keypoint if the tracking was unsuccessful.
    if (!tracking_success[keypoint_idx_k]) {
      continue;
    }

    // Drop keypoint if it moved too close to the image border as we can't compute a descriptor.
    const Eigen::Vector2d& point = tracked_keypoints_kp1[keypoint_idx_k];
    if (point(0) < kMinDistanceToImageBorderPx ||
        point(0) >= (camera_.imageWidth() - kMinDistanceToImageBorderPx) ||
        point(1) < kMinDistanceToImageBorderPx ||
        point(1) >= (camera_.imageHeight() - kMinDistanceToImageBorderPx)) {
      continue;
    }

    // Drop keypoint if it is marked for abortion.
    if (keypoint_indices_to_abort_.count(keypoint_idx_k) >= 1u) {
      continue;
    }

    // Drop keypoints that have moved too close to another tracked keypoint.
    occupancy_grid.addPointOrReplaceWeakestNearestPoints(
        WeightedKeypoint(point(1), point(0), -tracking_errors[keypoint_idx_k], keypoint_idx_k),
        kMinDistanceBetweenKeypointsPx);
  }
  timer_selection.Stop();

  // Set an infinite weight for all tracked keypoints in the occupancy grid to avoid replacing them
  // with new detected keypoints.
  occupancy_grid.setConstantWeightForAllPointsInGrid(
      std::numeric_limits<OccupancyGrid::WeightType>::max());

  // Detect new keypoints if the number of keypoints drops below the specified threshold.
  if (occupancy_grid.getNumPoints() < kMinFeatureCount) {
    // Create the detection mask consisting of the mask that prevents detecting points too close to
    // the image border and the mask of the current points in the occupancy grid.
    const size_t kMaxNumberOfKeypointPerCell = std::numeric_limits<size_t>::max();
    const cv::Mat detection_mask_occupancy_grid = occupancy_grid.getOccupancyMask(
        kMinDistanceBetweenKeypointsPx, kMaxNumberOfKeypointPerCell);
    cv::Mat detection_mask;
    cv::bitwise_and(detection_mask_image_border_, detection_mask_occupancy_grid, detection_mask);
    if (FLAGS_lk_show_detection_mask) {
      cv::namedWindow("detection mask");
      cv::imshow("detection mask", detection_mask);
      cv::waitKey(0);
    }

    // Detect new points.
    CHECK_LT(occupancy_grid.getNumPoints(), kMaxFeatureCount);
    const size_t num_keypoints_to_detect = kMaxFeatureCount - occupancy_grid.getNumPoints();

    Vector2dList new_keypoints;
    std::vector<double> new_keypoints_scores;
    detectNewKeypoints(frame_kp1->getRawImage(), num_keypoints_to_detect, detection_mask,
                       &new_keypoints, &new_keypoints_scores);

    // Add the new points to the occupancy grid. If a keypoint is inserted too close to an
    // existing point in the grid, the point with the higher score will be kept.
    // The grid stores an id for each point that corresponds to the keypoint index in the previous
    // frame for tracked keypoints. If it is a new detect keypoint the index -1 is set.
    const int kKeypointMatchIndexPreviousFrame = -1;
    for (size_t idx = 0u; idx < new_keypoints.size(); ++idx) {
      occupancy_grid.addPointOrReplaceWeakestNearestPoints(
          WeightedKeypoint(new_keypoints[idx](1), new_keypoints[idx](0),
                           new_keypoints_scores[idx], kKeypointMatchIndexPreviousFrame),
          kMinDistanceBetweenKeypointsPx);
    }
  }

  // Write the keypoints to the frame (k+1) in the following order [tracked, new keypoints]. Also
  // Extract the index-pairs between matching keypoints between frame (k) and (k+1).
  OccupancyGrid::PointList keypoints_kp1;
  occupancy_grid.getAllPointsInGrid(&keypoints_kp1);

  Vector2dList new_keypoints_kp1;
  new_keypoints_kp1.reserve(keypoints_kp1.size());
  size_t keypoint_idx_kp1 = 0u;
  for (const WeightedKeypoint& point : keypoints_kp1) {
    // Register a match if the point was successfully tracked from the previous frame. An id of -1
    // marks a new detected point.
    if (point.id >= 0) {
      const int keypoint_idx_k = point.id;
      matches_with_score_kp1_k->emplace_back(keypoint_idx_kp1, keypoint_idx_k, point.weight);
    }

    new_keypoints_kp1.emplace_back(point.v_cols, point.u_rows);
    ++keypoint_idx_kp1;
  }

  const double kKeypointUncertaintyPx = 0.8;
  insertAdditionalKeypointsToVisualFrame(new_keypoints_kp1, kKeypointUncertaintyPx, frame_kp1);

  // Reset the list of keypoints to abort tracking.
  keypoint_indices_to_abort_.clear();
  abort_keypoints_wrt_frame_id_.setInvalid();
}

void FeatureTrackerLk::trackKeypoints(const aslam::Quaternion& q_Ckp1_Ck,
                                      const aslam::VisualFrame& frame_k,
                                      const cv::Mat& image_frame_kp1,
                                      Vector2dList* tracked_keypoints_kp1,
                                      std::vector<unsigned char>* tracking_success,
                                      std::vector<float>* tracking_errors) const {
  aslam::timing::Timer timer_tracking("FeatureTrackerLk: track - trackKeypoints");
  CHECK_NOTNULL(tracked_keypoints_kp1)->clear();
  CHECK_NOTNULL(tracking_success)->clear();
  CHECK_NOTNULL(tracking_errors)->clear();

  // Early exit if the frame k does not contain any keypoints.
  if (!frame_k.hasKeypointMeasurements() || frame_k.getNumKeypointMeasurements() == 0u) {
    return;
  }

  // Predict the keypoint locations from the frame (k) to the frame (k+1) using the rotation prior.
  // The initial keypoint location is kept if the prediction failed.
  Eigen::Matrix2Xd predicted_keypoints_kp1;
  std::vector<unsigned char> prediction_success;
  predictKeypointsByRotation(frame_k, q_Ckp1_Ck, &predicted_keypoints_kp1, &prediction_success);

  // Convert the keypoint type to OpenCV.
  std::vector<cv::Point2f> keypoints_k, keypoints_kp1;
  convertKeypointVectorToCvPointList(frame_k.getKeypointMeasurements(), &keypoints_k);
  convertKeypointVectorToCvPointList(predicted_keypoints_kp1, &keypoints_kp1);

  // Find the keypoint location in the frame (k+1) starting from the predicted positions using
  // optical flow. If the flow wasn’t found, then the error is not defined. Use the
  // tracking_success parameter to find such cases.
  cv::calcOpticalFlowPyrLK(frame_k.getRawImage(),
                           image_frame_kp1,
                           keypoints_k,
                           keypoints_kp1,
                           *tracking_success,
                           *tracking_errors,
                           kWindowSize,
                           kMaxPyramidLevel,
                           kTerminationCriteria,
                           kOperationFlag,
                           kMinEigenThreshold);

  tracked_keypoints_kp1->reserve(keypoints_kp1.size());
  for (const cv::Point2f& tracked_point : keypoints_kp1) {
    tracked_keypoints_kp1->emplace_back(tracked_point.x, tracked_point.y);
  }

  CHECK_EQ(keypoints_k.size(), keypoints_kp1.size());
  CHECK_EQ(tracking_success->size(), keypoints_kp1.size());
  CHECK_EQ(tracking_errors->size(), keypoints_kp1.size());
  CHECK_EQ(tracked_keypoints_kp1->size(), keypoints_kp1.size());
}

void FeatureTrackerLk::detectNewKeypoints(const cv::Mat& image_kp1,
                                          size_t num_keypoints_to_detect,
                                          const cv::Mat& detection_mask,
                                          Vector2dList* keypoints,
                                          std::vector<double>* keypoint_scores) const {
  aslam::timing::Timer timer_detection("FeatureTrackerLk: detectNewKeypoints");
  CHECK_NOTNULL(keypoints)->clear();
  CHECK_NOTNULL(keypoint_scores)->clear();

  // Early exit if no keypoints need to be detected.
  if (num_keypoints_to_detect == 0u) {
    return;
  }

  // The detector needs to be reconstructed in each iteration as brisk doesn't provide an
  // interface to change the number of detected keypoints.
  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> detector(
      kBriskDetectorOctaves, kBriskDetectorUniformityRadius, kBriskDetectorAbsoluteThreshold,
      num_keypoints_to_detect);

  // Detect new keypoints in the unmasked image area.
  std::vector<cv::KeyPoint> keypoints_cv;
  keypoints_cv.reserve(num_keypoints_to_detect);
  detector.detect(image_kp1, keypoints_cv, detection_mask);

  // Convert the data types to the output structures.
  keypoints->reserve(keypoints_cv.size());
  keypoint_scores->reserve(keypoints_cv.size());
  for (size_t idx = 0u; idx < keypoints_cv.size(); ++idx) {
    // TODO(schneith): Brisk detector does not seem to respect the detection mask...
    if (keypoints_cv[idx].pt.x < kMinDistanceToImageBorderPx ||
        keypoints_cv[idx].pt.x >= (camera_.imageWidth() - kMinDistanceToImageBorderPx) ||
        keypoints_cv[idx].pt.y < kMinDistanceToImageBorderPx ||
        keypoints_cv[idx].pt.y >= (camera_.imageHeight() - kMinDistanceToImageBorderPx)) {
      continue;
    }

    keypoints->emplace_back(keypoints_cv[idx].pt.x, keypoints_cv[idx].pt.y);
    keypoint_scores->emplace_back(keypoints_cv[idx].response);
  }
}

void FeatureTrackerLk::swapKeypointIndicesToAbort(
    const aslam::FrameId& frame_id, std::unordered_set<size_t>* keypoint_indices_to_abort) {
  CHECK_NOTNULL(keypoint_indices_to_abort);
  CHECK(frame_id.isValid());
  keypoint_indices_to_abort_.swap(*keypoint_indices_to_abort);
  abort_keypoints_wrt_frame_id_ = frame_id;
}

}  // namespace aslam
