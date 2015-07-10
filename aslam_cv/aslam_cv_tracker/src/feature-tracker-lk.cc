#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/feature-tracker-lk.h>

#include <opencv/highgui.h>
namespace aslam {

FeatureTrackerLk::FeatureTrackerLk(const aslam::Camera& camera) {
  // Create the detection mask.
  detection_mask_ = cv::Mat::zeros(camera.imageHeight(), camera.imageWidth(), CV_8UC1);
  cv::Mat roi(detection_mask_, cv::Rect(kMinDistanceToImageBorderPx, kMinDistanceToImageBorderPx,
                                        camera.imageWidth() - 2 * kMinDistanceToImageBorderPx,
                                        camera.imageHeight() - 2 * kMinDistanceToImageBorderPx));
  roi = cv::Scalar(255);
}

void FeatureTrackerLk::track(const aslam::Quaternion& q_Ckp1_Ck,
                             const aslam::VisualFrame& frame_k,
                             aslam::VisualFrame* frame_kp1,
                             aslam::MatchesWithScore* matches_with_score_kp1_k) {
  CHECK_NOTNULL(frame_kp1);
  CHECK_NOTNULL(matches_with_score_kp1_k)->clear();

  // Make sure the nframes don't already contain keypoint/tracking information.
  CHECK(frame_kp1->hasRawImage());
  CHECK(!frame_kp1->hasKeypointMeasurements());
  CHECK(!frame_kp1->hasTrackIds());

  // Track existing feature of frame k to frame kp1.
  Vector2dList new_keypoints_kp1;
  new_keypoints_kp1.reserve(kMaxFeatureCount);
  if (frame_k.hasKeypointMeasurements() && frame_k.getNumKeypointMeasurements() > 0) {
    std::vector<cv::Point2f> keypoints_k;
    getKeypointsFromFrame(frame_k, &keypoints_k);

    // Use the rotation to predict keypoints in the next frame.
    Eigen::Matrix3Xd rays;
    Eigen::Matrix2Xd predicted_keypoints_kp1;
    std::vector<char> projection_successfull;
    std::vector<ProjectionResult> projection_result;
    frame_k.getCameraGeometry()->backProject3Vectorized(
        frame_k.getKeypointMeasurements(), &rays, &projection_successfull);
    rays = q_Ckp1_Ck.getRotationMatrix() * rays;
    frame_kp1->getCameraGeometry()->project3Vectorized(rays, &predicted_keypoints_kp1,
                                                       &projection_result);

    std::vector<cv::Point2f> tracked_keypoints_kp1;
    tracked_keypoints_kp1.reserve(keypoints_k.size());
    for (int idx = 0; idx < predicted_keypoints_kp1.cols(); ++idx) {
      if (projection_successfull[idx] && projection_result[idx].isKeypointVisible()) {
        tracked_keypoints_kp1.emplace_back(predicted_keypoints_kp1.col(idx)(0),
                                           predicted_keypoints_kp1.col(idx)(1));
      } else {
        // Default to no motion for prediction.
        tracked_keypoints_kp1.emplace_back(keypoints_k[idx]);
      }
    }

    // Output vector of errors. Each element of the vector is set to an error for the corresponding
    // feature. Type of the error measure can be set in flags parameter. If the flow wasn’t found
    // then the error is not defined (use the tracking_successful parameter to find such cases).
    std::vector<float> tracking_errors;
    tracking_errors.reserve(keypoints_k.size());

    // Output status vector (of unsigned chars). Each element of the vector is set to 1 if the flow
    // for the corresponding features has been found, otherwise, it is set to 0.
    std::vector<unsigned char> tracking_successful;
    tracking_successful.reserve(keypoints_k.size());

    cv::calcOpticalFlowPyrLK(frame_k.getRawImage(), frame_kp1->getRawImage(),
                             keypoints_k, tracked_keypoints_kp1, tracking_successful, tracking_errors,
                             kWindowSize, kMaxPyramidLevel, kTerminationCriteria, kOperationFlag,
                             kMinEigenThreshold);

    // Sort indices by tracking error.
    std::vector<size_t> indices_sorted_by_ascending_tracking_error(tracking_errors.size());
    for (size_t i = 0u; i < indices_sorted_by_ascending_tracking_error.size(); ++i) {
      indices_sorted_by_ascending_tracking_error[i] = i;
    }
    std::sort(indices_sorted_by_ascending_tracking_error.begin(),
              indices_sorted_by_ascending_tracking_error.end(),
              [&tracking_errors](size_t index_1, size_t index_2) {
      return tracking_errors[index_1] < tracking_errors[index_2];
    });
    CHECK_EQ(keypoints_k.size(), tracked_keypoints_kp1.size());

    Eigen::Matrix<size_t, kGridCellResolution, kGridCellResolution> occupancy_grid;
    if (kUseOccupancyGrid) {
      occupancy_grid.setZero();
    }

    // Make sure the registered keypoints to abort belong to this frame.
    if (!keypoint_indices_to_abort_.empty() && abort_keypoints_wrt_frame_id_ != frame_k.getId()) {
      LOG(WARNING) << "Keypoints to abort do not match the processed frame.";
      keypoint_indices_to_abort_.clear();
    }

    size_t keypoint_idx_kp1 = 0u;
    for (size_t j = 0u; j < keypoints_k.size(); ++j) {
      const size_t keypoint_idx_k = indices_sorted_by_ascending_tracking_error[j];

      // Drop tracks that are marked for abortion or invalid.
      if (keypoint_indices_to_abort_.count(keypoint_idx_k) == 1u) {
        continue;
      }

      // Drop tracks where the tracking failed or that are close to the border as we can't
      // compute descriptors.
      if (tracked_keypoints_kp1[keypoint_idx_k].x < kMinDistanceToImageBorderPx ||
          tracked_keypoints_kp1[keypoint_idx_k].x >=
            (frame_k.getRawImage().cols - kMinDistanceToImageBorderPx) ||
          tracked_keypoints_kp1[keypoint_idx_k].y < kMinDistanceToImageBorderPx ||
          tracked_keypoints_kp1[keypoint_idx_k].y >=
            (frame_k.getRawImage().rows - kMinDistanceToImageBorderPx) ||
          !tracking_successful[keypoint_idx_k]) {
        continue;
      }

      // Drop tracks if there are too many landmarks already in the the bucket.
      if (kUseOccupancyGrid) {
        const size_t image_width = frame_k.getRawImage().cols;
        const size_t image_height = frame_k.getRawImage().rows;
        size_t grid_x = std::floor(
            tracked_keypoints_kp1[keypoint_idx_k].x * kGridCellResolution / image_width );
        size_t grid_y = std::floor(
            tracked_keypoints_kp1[keypoint_idx_k].y * kGridCellResolution / image_height );

        if (occupancy_grid(grid_x, grid_y) >= kMaxLandmarksPerCell) {
          continue;
        }
        ++occupancy_grid(grid_x, grid_y);
      }

      // Index k corresponds to current frame kp1, index i corresponds to previous frame k.
      matches_with_score_kp1_k->emplace_back(keypoint_idx_kp1, keypoint_idx_k,
                                             -tracking_errors[keypoint_idx_k]);
      new_keypoints_kp1.emplace_back(tracked_keypoints_kp1[keypoint_idx_k].x,
                                     tracked_keypoints_kp1[keypoint_idx_k].y);

      CHECK_LT(keypoint_idx_kp1, tracked_keypoints_kp1.size());
      ++keypoint_idx_kp1;
    }
  }

  // Initialize new features if the number of tracked features drops below a certain threshold or
  // there aren't any during initialization. They will be tracked in next time step.
  if (!frame_kp1->hasKeypointMeasurements() ||
      frame_kp1->getNumKeypointMeasurements() < kMinFeatureCount) {
    Vector2dList detected_keypoints;
    detectGfttCorners(frame_kp1->getRawImage(), &detected_keypoints);

    if (kUseOccupancyGrid) {
      // Decide which new features to track based on occupancy grid.
      Vector2dList detected_keypoints_in_grid;
      occupancyGrid(*frame_kp1, detected_keypoints, &detected_keypoints_in_grid);

      new_keypoints_kp1.insert(new_keypoints_kp1.end(), detected_keypoints_in_grid.begin(),
                           detected_keypoints_in_grid.end());
    } else {
      // Add ALL features to the frame.
      new_keypoints_kp1.insert(new_keypoints_kp1.end(), detected_keypoints.begin(),
                           detected_keypoints.end());
    }
  }
  CHECK(!new_keypoints_kp1.empty());

  // Write the tracked and new keypoints to the frame kp1.
  insertAdditionalKeypointsToFrame(new_keypoints_kp1, frame_kp1);
  CHECK(frame_kp1->hasKeypointMeasurements());

  keypoint_indices_to_abort_.clear();
  abort_keypoints_wrt_frame_id_.setInvalid();
}

void FeatureTrackerLk::detectGfttCorners(const cv::Mat& image, Vector2dList* detected_keypoints) {
  CHECK_NOTNULL(detected_keypoints);
  std::vector<cv::Point2f> detected_keypoints_cv;
  cv::goodFeaturesToTrack(image, detected_keypoints_cv, kMaxFeatureCount,
                          kGoodFeaturesToTrackQualityLevel, kGoodFeaturesToTrackMinDistancePixel,
                          detection_mask_);

  // Subpixel refinement of detected corners.
  cv::cornerSubPix(image, detected_keypoints_cv, kSubPixelWinSize, kSubPixelZeroZone,
                   kTerminationCriteria);

  // Simple type conversion and removal of tracks that are close to the border as we can't compute
  // descriptors.
  detected_keypoints->clear();
  for (size_t i = 0u; i < detected_keypoints_cv.size(); ++i) {
    if (detected_keypoints_cv[i].x < kMinDistanceToImageBorderPx ||
        detected_keypoints_cv[i].x >= (detection_mask_.cols - kMinDistanceToImageBorderPx) ||
        detected_keypoints_cv[i].y < kMinDistanceToImageBorderPx ||
        detected_keypoints_cv[i].y >= (detection_mask_.rows - kMinDistanceToImageBorderPx)) {
      continue;
    }
    detected_keypoints->emplace_back(detected_keypoints_cv[i].x, detected_keypoints_cv[i].y);
  }
}

// TODO(hitimo): Simplify as soon as new visual pipeline is ready.
void FeatureTrackerLk::insertAdditionalKeypointsToFrame(const Vector2dList& new_keypoint_list,
                                                        aslam::VisualFrame* frame) {
  CHECK_NOTNULL(frame);
  const size_t num_new_keypoints = new_keypoint_list.size();
  Eigen::Matrix2Xd new_keypoints(2, num_new_keypoints);
  for (size_t i = 0; i < num_new_keypoints; ++i) {
    new_keypoints.col(i) = new_keypoint_list[i];
  }

  const double kKeypointUncertaintyPx = 0.8;
  if (frame->hasKeypointMeasurements()) {
    CHECK(frame->hasTrackIds());
    CHECK(frame->hasKeypointMeasurementUncertainties());
    const size_t old_size = frame->getNumKeypointMeasurements();
    const size_t extended_size = old_size + num_new_keypoints;

    // Resize the existing structures.
    Eigen::Matrix2Xd* keypoints = CHECK_NOTNULL(frame->getKeypointMeasurementsMutable());
    Eigen::VectorXi* track_ids = CHECK_NOTNULL(frame->getTrackIdsMutable());
    Eigen::VectorXd* uncertainties =
        CHECK_NOTNULL(frame->getKeypointMeasurementUncertaintiesMutable());
    CHECK_EQ(keypoints->cols(), track_ids->rows());

    keypoints->conservativeResize(Eigen::NoChange, extended_size);
    track_ids->conservativeResize(extended_size);
    uncertainties->conservativeResize(extended_size);

    // Add the new values.
    keypoints->block(0, old_size, 2, num_new_keypoints) = new_keypoints;
    track_ids->segment(old_size, num_new_keypoints).setConstant(-1);
    uncertainties->segment(old_size, num_new_keypoints).setConstant(kKeypointUncertaintyPx);

    CHECK_EQ(static_cast<int>(extended_size), frame->getKeypointMeasurements().cols());
    CHECK_EQ(static_cast<int>(extended_size), frame->getKeypointMeasurementUncertainties().rows());
    CHECK_EQ(static_cast<int>(extended_size), frame->getTrackIds().rows());
  } else {
    // Just swap in the keypoints, set invalid track ids and a constant measurement uncertainty.
    frame->setKeypointMeasurements(new_keypoints);

    Eigen::VectorXi track_ids(num_new_keypoints);
    track_ids.setConstant(-1);
    frame->swapTrackIds(&track_ids);

    Eigen::VectorXd uncertainties(num_new_keypoints);
    uncertainties.setConstant(kKeypointUncertaintyPx);
    frame->swapKeypointMeasurementUncertainties(&uncertainties);
  }
}

void FeatureTrackerLk::getKeypointsFromFrame(const aslam::VisualFrame& frame,
                                             std::vector<cv::Point2f>* keypoints_out) {
  CHECK_NOTNULL(keypoints_out);
  const Eigen::Matrix2Xd& keypoints = frame.getKeypointMeasurements();
  keypoints_out->reserve(keypoints.cols());
  for (size_t i = 0u; i < static_cast<size_t>(keypoints.cols()); ++i) {
    keypoints_out->emplace_back(keypoints.col(i)(0), keypoints.col(i)(1));
  }
}

void FeatureTrackerLk::occupancyGrid(const aslam::VisualFrame& frame,
                                     const Vector2dList& detected_keypoints,
                                     Vector2dList* detected_keypoints_in_grid) {
  CHECK_NOTNULL(detected_keypoints_in_grid);
  Eigen::Matrix<size_t, kGridCellResolution, kGridCellResolution> occupancy_grid;
  occupancy_grid.setZero();
  const size_t image_width = frame.getRawImage().cols;
  const size_t image_height = frame.getRawImage().rows;

  size_t num_previous_keypoints;
  Eigen::Matrix2Xd previous_keypoints;
  if (frame.hasKeypointMeasurements()) {
    previous_keypoints = frame.getKeypointMeasurements();
    num_previous_keypoints = frame.getNumKeypointMeasurements();
  } else {
    num_previous_keypoints = 0;
  }

  // First extract the keypoints that are currently tracked.
  for (size_t keypoint_idx = 0; keypoint_idx < num_previous_keypoints; ++keypoint_idx) {
    const Eigen::Vector2d previous_keypoint = previous_keypoints.col(keypoint_idx);
    CHECK_GE(previous_keypoint(0), 0u);
    CHECK_GE(previous_keypoint(1), 0u);

    const Eigen::Matrix<size_t, 2, 1> grid_cell(
        std::floor(previous_keypoint(0) / image_width * kGridCellResolution),
        std::floor(previous_keypoint(1) / image_height * kGridCellResolution));
    if (grid_cell(0) >= kGridCellResolution) {
      LOG(FATAL) << "FAIL grid_cell_x " << grid_cell(0)
                 << " point " << previous_keypoint(0) << " " << previous_keypoint(1);
    }
    if (grid_cell(1) >= kGridCellResolution) {
      LOG(FATAL) << "FAIL grid_cell_y " << grid_cell(1)
                 << " point " << previous_keypoint(0) << " " << previous_keypoint(1);
    }
    occupancy_grid(grid_cell(0), grid_cell(1)) += 1;
  }

  // Then add the new (detected) keypoints - if possible.
  detected_keypoints_in_grid->clear();
  for (size_t keypoint_idx = 0; keypoint_idx < detected_keypoints.size(); ++keypoint_idx) {
      const Eigen::Vector2d point(detected_keypoints[keypoint_idx]);
      const Eigen::Matrix<size_t, 2, 1> grid_cell(
          std::floor(point(0) / image_width * kGridCellResolution),
          std::floor(point(1) / image_height * kGridCellResolution));
    if (grid_cell(0) >= kGridCellResolution || grid_cell(1) >= kGridCellResolution) {
      continue;
    }
    if (occupancy_grid(grid_cell(0), grid_cell(1)) < kMaxLandmarksPerCell) {
      occupancy_grid(grid_cell(0), grid_cell(1)) += 1;
      detected_keypoints_in_grid->push_back(point);
    }
  }
}

}  // namespace aslam
