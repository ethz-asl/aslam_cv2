#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/feature-tracker-lk.h>

#include <opencv/highgui.h>
namespace aslam {

FeatureTrackerLk::FeatureTrackerLk(const aslam::Camera& camera)
    : first_frame_processed_(false) {
  // Create the detection mask.
  detection_mask_ = cv::Mat::zeros(camera.imageHeight(), camera.imageWidth(), CV_8UC1);
  cv::Mat roi(detection_mask_, cv::Rect(kMinDistanceToImageBorderPx, kMinDistanceToImageBorderPx,
                                        camera.imageWidth() - 2 * kMinDistanceToImageBorderPx,
                                        camera.imageHeight() - 2 * kMinDistanceToImageBorderPx));
  roi = cv::Scalar(255);
}

void FeatureTrackerLk::track(const aslam::VisualFrame::Ptr& frame_kp1,
                             const aslam::VisualFrame::Ptr& frame_k,
                             const aslam::Quaternion& q_Ckp1_Ck,
                             aslam::MatchesWithScore* matches_with_score_kp1_k) {
  CHECK(frame_k);
  CHECK(frame_kp1);
  CHECK_NOTNULL(matches_with_score_kp1_k);

  // Remove the possibly available BRISK information in the frames.
  // TODO(schneith): This can be removed once the VisualPipeline is refactored.
  frame_kp1->clearKeypointChannels();

  if (!first_frame_processed_) {
    frame_k->clearKeypointChannels();
    // Detect new features.
    Vector2dList detected_keypoints;
    detectGfttCorners(frame_kp1->getRawImage(), &detected_keypoints);
    if (kUseOccupancyGrid) {
      // Decide which new features to track based on occupancy grid.
      Vector2dList detected_keypoints_in_grid;
      occupancyGrid(frame_kp1, detected_keypoints, &detected_keypoints_in_grid);

      // Add these features to the frame.
      insertAdditionalKeypointsToFrame(detected_keypoints_in_grid, frame_kp1);
    } else {
      // Add ALL features to the frame.
      insertAdditionalKeypointsToFrame(detected_keypoints, frame_kp1);
    }
    // This is the first frame. We are done here.
    first_frame_processed_ = true;
    return;
  }

  // Track the features in the current frame kp1.
  Vector2dList new_keypoints;
  // Check if there are features to track at all.
  if (frame_k->getNumKeypointMeasurements() > 0) {
    // Output status vector (of unsigned chars). Each element of the vector is set to 1 if the flow
    // for the corresponding features has been found, otherwise, it is set to 0.
    std::vector<unsigned char> tracking_successful;

    // Output vector of errors. Each element of the vector is set to an error for the corresponding
    // feature. Type of the error measure can be set in flags parameter. If the flow wasn’t found
    // then the error is not defined (use the tracking_successful parameter to find such cases).
    std::vector<float> tracking_error;

    std::vector<cv::Point2f> keypoints_k, keypoints_kp1;
    getKeypointsfromFrame(frame_k, &keypoints_k);

    // Use the rotation to predict keypoints in the next frame.
    Eigen::Matrix3Xd rays;
    Eigen::Matrix2Xd predicted_keypoints;
    std::vector<bool> projection_successfull;
    std::vector<ProjectionResult> projection_result;
    frame_k->getCameraGeometry()->backProject3Vectorized(
        frame_k->getKeypointMeasurements(), &rays, &projection_successfull);
    rays = q_Ckp1_Ck.getRotationMatrix() * rays;
    frame_kp1->getCameraGeometry()->project3Vectorized(rays,
                                                       &predicted_keypoints,
                                                       &projection_result);
    keypoints_kp1.reserve(keypoints_k.size());
    for (int i = 0; i < predicted_keypoints.cols(); ++i) {
      if (projection_successfull[i]
          && projection_result[i].isKeypointVisible()) {
        keypoints_kp1.emplace_back(predicted_keypoints.col(i)(0),
                                   predicted_keypoints.col(i)(1));
      } else {
        // Default to no motion for prediction.
        keypoints_kp1.emplace_back(keypoints_k[i]);
      }
    }

    cv::calcOpticalFlowPyrLK(frame_k->getRawImage(), frame_kp1->getRawImage(),
                             keypoints_k, keypoints_kp1, tracking_successful, tracking_error,
                             kWindowSize, kMaxPyramidLevel, kTerminationCriteria, kOperationFlag,
                             kMinEigenThreshold);

    // Sort indices by tracking error.
    std::vector<size_t> ind(tracking_error.size());
    for (size_t i = 0; i < ind.size(); ++i) ind[i] = i;
    std::sort(ind.begin(), ind.end(), [&tracking_error](size_t i1, size_t i2) {
      return tracking_error[i1] < tracking_error[i2];
    });

    Eigen::Matrix<size_t, kGridCellResolution, kGridCellResolution> occupancy_grid;
    occupancy_grid.setZero();
    const size_t image_width = frame_k->getRawImage().cols;
    const size_t image_height = frame_k->getRawImage().rows;

    matches_with_score_kp1_k->clear();
    CHECK_EQ(keypoints_k.size(), keypoints_kp1.size());
    size_t k = 0;
    for (size_t j = 0; j < keypoints_k.size(); ++j) {
      size_t i = ind[j]; // Get sorted index.

      if (keypoints_kp1[i].x < kMinDistanceToImageBorderPx ||
          keypoints_kp1[i].x >= (frame_k->getRawImage().cols - kMinDistanceToImageBorderPx) ||
          keypoints_kp1[i].y < kMinDistanceToImageBorderPx ||
          keypoints_kp1[i].y >= (frame_k->getRawImage().rows - kMinDistanceToImageBorderPx) ||
          !tracking_successful[i]) {
        continue;
      }

      {
        size_t grid_x = std::floor(keypoints_kp1[i].x * kGridCellResolution / image_width );
        size_t grid_y = std::floor(keypoints_kp1[i].y * kGridCellResolution / image_height );

        if (occupancy_grid(grid_x, grid_y) >= kMaxLandmarksPerCell) {
          // Don't use the track if there are too many landmarks in this cell.
          continue;
        }
        ++occupancy_grid(grid_x, grid_y);
       }

      new_keypoints.emplace_back(keypoints_kp1[i].x, keypoints_kp1[i].y);

      // Create the matches.
      // Index k corresponds to current frame kp1, index i corresponds to previous frame k.
      matches_with_score_kp1_k->emplace_back(k, i, -tracking_error[i]);
      CHECK_LT(k, keypoints_kp1.size());
      CHECK_LT(i, keypoints_k.size());
      ++k;
    }
  }

  // Add the keypoints that were tracked from frame k to frame k+1
  // to the current frame k+1.
  if (!new_keypoints.empty()) {
    insertAdditionalKeypointsToFrame(new_keypoints, frame_kp1);
  }

  // If the number of tracked features drops below threshold, then add
  // new features to the current frame. They will be tracked in next
  // time step.
  if (frame_kp1->getNumKeypointMeasurements() < kMinFeatureCount) {
    // Detect new features.
    Vector2dList detected_keypoints;
    detectGfttCorners(frame_kp1->getRawImage(), &detected_keypoints);

    if (kUseOccupancyGrid) {
      // Decide which new features to track based on occupancy grid.
      Vector2dList detected_keypoints_in_grid;
      occupancyGrid(frame_kp1, detected_keypoints, &detected_keypoints_in_grid);

      // Add these features to the frame.
      insertAdditionalKeypointsToFrame(detected_keypoints_in_grid, frame_kp1);
    } else {
      // Add ALL features to the frame.
      insertAdditionalKeypointsToFrame(detected_keypoints, frame_kp1);
    }
  }
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

  // Simple type conversion.
  detected_keypoints->clear();
  for (size_t i = 0; i < detected_keypoints_cv.size(); ++i) {
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
                                                        aslam::VisualFrame::Ptr frame) {
  CHECK(frame);
  size_t old_size = frame->getNumKeypointMeasurements();
  size_t num_new_keypoints = new_keypoint_list.size();
  size_t extended_size = old_size + num_new_keypoints;

  Eigen::Matrix2Xd new_keypoints(2, num_new_keypoints);
  for (size_t i = 0; i < num_new_keypoints; ++i) {
    new_keypoints.col(i) = new_keypoint_list[i];
  }

  const double kKeypointUncertaintyPx = 0.8;
  if (frame->hasKeypointMeasurements()) {
    CHECK(frame->hasTrackIds());

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
  CHECK_EQ(static_cast<int>(extended_size), frame->getKeypointMeasurements().cols());
  CHECK_EQ(static_cast<int>(extended_size), frame->getKeypointMeasurementUncertainties().rows());
  CHECK_EQ(static_cast<int>(extended_size), frame->getTrackIds().rows());
}

void FeatureTrackerLk::getKeypointsfromFrame(const aslam::VisualFrame::Ptr frame,
                                      std::vector<cv::Point2f>* keypoints_out) {
  CHECK(frame);
  CHECK_NOTNULL(keypoints_out);
  const Eigen::Matrix2Xd& keypoints = frame->getKeypointMeasurements();
  for (size_t i = 0u; i < static_cast<size_t>(keypoints.cols()); ++i) {
    keypoints_out->emplace_back(keypoints.col(i)(0), keypoints.col(i)(1));
  }
}

void FeatureTrackerLk::occupancyGrid(const aslam::VisualFrame::Ptr frame,
                                     const Vector2dList& detected_keypoints,
                                     Vector2dList* detected_keypoints_in_grid) {
  CHECK(frame);
  CHECK_NOTNULL(detected_keypoints_in_grid);
  Eigen::Matrix<size_t, kGridCellResolution, kGridCellResolution> occupancy_grid;
  occupancy_grid.setZero();
  const size_t image_width = frame->getRawImage().cols;
  const size_t image_height = frame->getRawImage().rows;

  size_t num_previous_keypoints;
  Eigen::Matrix2Xd previous_keypoints;
  if (frame->hasKeypointMeasurements()) {
    previous_keypoints = frame->getKeypointMeasurements();
    num_previous_keypoints = frame->getNumKeypointMeasurements();
  } else {
    num_previous_keypoints = 0;
  }

  // First extract the keypoints that are are currently tracked.
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
