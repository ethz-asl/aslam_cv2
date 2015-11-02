#ifndef ASLAM_FEATURE_TRACKER_LK_H_
#define ASLAM_FEATURE_TRACKER_LK_H_

#include <unordered_set>
#include <vector>

#include <aslam/cameras/camera.h>
#include <aslam/common/memory.h>
#include <aslam/matcher/match.h>
#include <aslam/tracker/feature-tracker.h>
#include <aslam/common/occupancy-grid.h>
#include <Eigen/Core>
#include <gflags/gflags.h>
#include <opencv2/video/tracking.hpp>

namespace aslam {
class VisualFrame;

struct LkTrackerSettings {
  /// Brisk Harris detector settings.
  size_t brisk_detector_octaves;
  size_t brisk_detector_uniformity_radius_px;
  size_t brisk_detector_absolute_threshold;

  /// Min. distance between the detected keypoints.
  double min_distance_between_features_px;
  /// Maximum number of keypoint to detect.
  size_t max_feature_count;
  /// Threshold when to detect new keypoints.
  size_t min_feature_count;

  /// The algorithm calculates the minimum eigen value of a 2x2 normal matrix of optical flow
  /// equations (this matrix is called a spatial gradient matrix in [Bouguet00]), divided by number
  /// of pixels in a window; if this value is less than kMinEigThreshold, then a corresponding
  /// feature is filtered out and its flow is not processed, so it allows to remove bad points and
  /// get a performance boost.
  double lk_min_eigen_threshold;
  /// Maximal pyramid level number. If set to 0, pyramids are not used (single level), if set to 1,
  /// two levels are used, and so on.
  size_t lk_max_pyramid_level;
  /// Size of the search window at each pyramid level.
  size_t lk_window_size;

  LkTrackerSettings();
};

class FeatureTrackerLk : public FeatureTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef aslam::Aligned<std::vector, Eigen::Vector2d>::type Vector2dList;
  typedef aslam::WeightedKeypoint<double, double, int> WeightedKeypoint;
  typedef aslam::WeightedOccupancyGrid<WeightedKeypoint> OccupancyGrid;

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{
 public:
  FeatureTrackerLk(const aslam::Camera& camera, const LkTrackerSettings& settings);
  virtual ~FeatureTrackerLk() {}

 private:
  void initialize(const aslam::Camera& camera);
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Tracking related methods.
  /// @{
 public:
  /// External interface for feature tracking. This method tracks existig keypoints from frame (k)
  /// to frame (k+1) and initializes new keypoints if the number of keypoints drops below the
  /// specified threshold.
  virtual void track(
      const aslam::Quaternion& q_Ckp1_Ck, const aslam::VisualFrame& frame_k,
      aslam::VisualFrame* frame_kp1,
      aslam::MatchesWithScore* matches_with_score_kp1_k) override;

  /// Takes a visual frame with no keypoints, and initializes new keypoints.
  /// Uses the class settings and an occupancy grid.
  void initializeKeypointsInEmptyVisualFrame(aslam::VisualFrame* frame) const;

 private:
  /// Track existing keypoints from frame (k) to frame (k+1). Make sure that the ordering of the
  /// keypoints in the keypoints_kp1 remains unchanged when writing the keypoints to the keypoint
  /// channel of the VisualFrame.
  void trackKeypoints(const aslam::Quaternion& q_Ckp1_Ck,
                      const aslam::VisualFrame& frame_k,
                      const cv::Mat& image_frame_kp1,
                      Vector2dList* tracked_keypoints_kp1,
                      std::vector<unsigned char>* tracking_success,
                      std::vector<float>* tracking_errors) const;

  /// Detects new keypoints in the given visual frame, using the given detection
  /// mask and
  /// the given occupancy grid.
  /// The keypoints will not be added to the visual frame, but only to the given
  /// occupancy grid.
  void detectNewKeypointsInVisualFrame(const aslam::VisualFrame& frame,
                                       const cv::Mat& detection_mask,
                                       OccupancyGrid* occupancy_grid) const;

  /// Operation flag. See opencv documentation for details.
  static constexpr size_t kOperationFlag = cv::OPTFLOW_USE_INITIAL_FLOW;// ||
      //cv::OPTFLOW_LK_GET_MIN_EIGENVALS;

  /// Size of the search window at each pyramid level.
  const cv::Size lk_window_size_;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Keypoint detection related methods.
  /// @{
 private:
  /// Detect new keypoints in the specified image.
  void detectNewKeypoints(const cv::Mat& image_kp1,
                          size_t num_keypoints_to_detect,
                          const cv::Mat& detection_mask,
                          Vector2dList* keypoints,
                          std::vector<double>* keypoint_scores) const;

  /// Enforce a minimal distance of all keypoints to the image border.
  const size_t kMinDistanceToImageBorderPx = 30u;

  /// Parameter specifying the termination criteria of the iterative search algorithm
  /// (after the specified maximum number of iterations criteria.maxCount or when the search
  /// window moves by less than criteria.epsilon).
  const cv::TermCriteria kTerminationCriteria = cv::TermCriteria(
      cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Method to abort the tracking of keypoints.
  /// @{
 public:
  /// Set a list of keypoint ids that have been identified as outliers in the last update step.
  /// The tracking of these features will be aborted.
  virtual void swapKeypointIndicesToAbort(
      const aslam::FrameId& frame_id, std::unordered_set<size_t>* keypoint_indices_to_abort);

  /// Keypoint indices wrt. to the last frame for which the tracking should be aborted during
  /// the next call to track().
  std::unordered_set<size_t> keypoint_indices_to_abort_;
  aslam::FrameId abort_keypoints_wrt_frame_id_;

  /// @}

 private:
  const aslam::Camera& camera_;
  const LkTrackerSettings settings_;

  /// Detection mask that prevents detecting keypoints close to the image border.
  cv::Mat detection_mask_image_border_;
};
}  // namespace aslam

#endif  // ASLAM_FEATURE_TRACKER_LK_H_
