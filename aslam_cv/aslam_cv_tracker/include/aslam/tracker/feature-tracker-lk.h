#ifndef ASLAM_FEATURE_TRACKER_LK_H_
#define ASLAM_FEATURE_TRACKER_LK_H_

#include <unordered_set>
#include <vector>

#include <aslam/common/memory.h>
#include <aslam/matcher/match.h>
#include <aslam/tracker/feature-tracker.h>
#include <brisk/brisk.h>
#include <Eigen/Dense>
#include <opencv2/video/tracking.hpp>

namespace aslam {
class VisualFrame;

class FeatureTrackerLk : public FeatureTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef aslam::Aligned<std::vector, Eigen::Vector2d>::type Vector2dList;

  FeatureTrackerLk(const aslam::Camera& camera);
  virtual ~FeatureTrackerLk() {}

  /// \brief Main Lk-tracker routine.
  /// @param[in] q_Ckp1_Ck Rotation taking points from the Ck frame to the Ckp1 frame.
  /// @param[in] frame_k Frame at time step k.
  /// @param[in,out] frame_kp1 Frame at time step k+1.
  /// @param[out] matches_with_score_kp1_k Detected matches from frame k to frame k+1.
  virtual void track(const aslam::Quaternion& q_Ckp1_Ck,
                     const aslam::VisualFrame& frame_k,
                     aslam::VisualFrame* frame_kp1,
                     aslam::MatchesWithScore* matches_with_score_kp1_k);

  /// Set a list of keypoint ids that have been identified as outliers in the last update step.
  /// The tracking of these features will be aborted.
  virtual void swapKeypointIndicesToAbort(
      const aslam::FrameId& frame_id, std::unordered_set<size_t>* keypoint_indices_to_abort) {
    CHECK_NOTNULL(keypoint_indices_to_abort);
    CHECK(frame_id.isValid());
    keypoint_indices_to_abort_.swap(*keypoint_indices_to_abort);
    abort_keypoints_wrt_frame_id_ = frame_id;
  }

 private:
  /// \brief Detect good features to track.
  /// @param[in] image Extract features from this image.
  /// @param[out] detected_keypoints List of detected keypoints.
  void detectGfttCorners(const cv::Mat& image, Vector2dList* detected_keypoints);

  /// \brief Apply keypoints of type Eigen::Vector2d to frame.
  /// @param[in] new_keypoints Keypoints_new to be applied to the frame.
  /// @param[in,out] frame_kp1 Frame to which the new keypoints should be applied.
  void insertAdditionalKeypointsToFrame(const Vector2dList& new_keypoints,
                                        aslam::VisualFrame* frame_kp1);

  /// \brief Get keypoints of type cv::Point2f from frame.
  /// @param[in] frame Frame from which the keypoints should be extracted.
  /// @param[out] keypoints_out Keypoints extracted from frame.
  void getKeypointsFromFrame(const aslam::VisualFrame& frame,
                             std::vector<cv::Point2f>* keypoints_out);

  /// \brief Build up an occupancy grid and only add new features to empty cells.
  /// @param[in] frame Frame from which keypoints are extracted. The keypoints are
  /// then inserted into the occupancy grid.
  /// @param[in] detected_keypoints Points we want to add to the occupancy grid.
  /// @param[out] detected_keypoints_in_grid Points actually added based on occupancy grid.
  void occupancyGrid(const aslam::VisualFrame& frame, const Vector2dList& detected_keypoints,
                     Vector2dList* detected_keypoints_in_grid);

  //////////////////////////////////////////////////////////////
  /// \name Parameters for Feature Detection.
  /// @{
  /// Parameter characterizing the minimal accepted quality of image corners. The parameter value
  /// is multiplied by the best corner quality measure, which is the minimal eigenvalue
  /// (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners
  /// with the quality measure less than the product are rejected. For example, if the best corner
  /// has the quality measure = 1500, and the qualityLevel=0.01, then all the corners with the
  /// quality measure less than 15 are rejected.
  static constexpr double kGoodFeaturesToTrackQualityLevel = 0.01;

  /// Minimum possible euclidean distance between the returned corners.
  static constexpr double kGoodFeaturesToTrackMinDistancePixel = 5.0;

  /// Maximum number of corners to return. If there are more corners than are found,
  /// the strongest of them are returned.
  static constexpr size_t kMaxFeatureCount = 750u;

  /// Minimum number of features to be tracked. Initialize new features to track if
  /// number is/drops below this number.
  static constexpr size_t kMinFeatureCount = 500u;

  /// Parameter for corner refinement. Half of the side length of the search window.
  /// For example, if kSubPixWinSize=Size(5,5), then a 5*2+1 \times 5*2+1 = 11 \times 11
  /// search window is used.
  const cv::Size kSubPixelWinSize = cv::Size(10, 10);

  /// Half of the size of the dead region in the middle of the search zone over which the
  /// summation in the sub-pixel formula is not done. It is used sometimes to avoid possible
  /// singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there
  /// is no such a size.
  const cv::Size kSubPixelZeroZone = cv::Size(-1, -1);

  /// Grid cell resolution of the occupancy grid.
  static const size_t kGridCellResolution = 16u;

  /// Maximum number of landmarks per cell in occupancy grid.
  static const size_t kMaxLandmarksPerCell = 4u;

  /// Either use occupancy grid and SimpleTrackManager or
  /// do not use occupancy grid and do use UniformTrackManager!
  static const bool kUseOccupancyGrid = true;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Parameters for Feature Tracking.
  /// @{
  /// The algorithm calculates the minimum eigen value of a 2x2 normal matrix of optical flow
  /// equations (this matrix is called a spatial gradient matrix in [Bouguet00]), divided by number
  /// of pixels in a window; if this value is less than kMinEigThreshold, then a corresponding
  /// feature is filtered out and its flow is not processed, so it allows to remove bad points and
  /// get a performance boost.
  static constexpr double kMinEigenThreshold = 0.001;

  /// 0-based maximal pyramid level number. If set to 0, pyramids are not used (single level),
  /// if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm
  /// will use as many levels as pyramids have but no more than maxLevel.
  static constexpr size_t kMaxPyramidLevel = 2u;

  /// Operation flag. See opencv documentation for details.
  static constexpr size_t kOperationFlag = cv::OPTFLOW_USE_INITIAL_FLOW;

  /// Parameter specifying the termination criteria of the iterative search algorithm
  /// (after the specified maximum number of iterations criteria.maxCount or when the search
  /// window moves by less than criteria.epsilon.
  const cv::TermCriteria kTerminationCriteria = cv::TermCriteria(
      cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);

  /// Size of the search window at each pyramid level.
  const cv::Size kWindowSize = cv::Size(21, 21);

  /// Enforce a minimal distance of all keypoints to the image border.
  const size_t kMinDistanceToImageBorderPx = 30u;
  /// @}

  /// Mask the area where no tracks should be spawned.
  cv::Mat detection_mask_;

  /// Keypoint indices wrt. to the last frame for which the tracking should be aborted during
  /// the next call to track().
  std::unordered_set<size_t> keypoint_indices_to_abort_;
  aslam::FrameId abort_keypoints_wrt_frame_id_;

  /// Optional brisk harris detector.
  size_t kBriskOctaves = 0;
  size_t kBriskUniformityRadius = 0;
  size_t kBriskAbsoluteThreshold = 15;
  std::unique_ptr<brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>> detector_;
};
}  // namespace aslam

#endif  // ASLAM_FEATURE_TRACKER_LK_H_
