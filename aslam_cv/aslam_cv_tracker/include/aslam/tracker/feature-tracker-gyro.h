#ifndef ASLAM_GYRO_TRACKER_H_
#define ASLAM_GYRO_TRACKER_H_

#include <array>
#include <deque>
#include <memory>
#include <vector>

#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/macros.h>
#include <aslam/tracker/feature-tracker.h>

namespace aslam {
class VisualFrame;
class Camera;

struct GyroTrackerSettings {
  GyroTrackerSettings();

  double lk_max_num_candidates_ratio_kp1;
  size_t lk_max_status_track_length;

  // calcOpticalFlowPyrLK parameters:
  cv::TermCriteria lk_termination_criteria;
  cv::Size lk_window_size;
  int lk_max_pyramid_levels;
  int lk_operation_flag;
  double lk_min_eigenvalue_threshold;

  // Keypoint uncertainty. It's quite arbitrary...
  static constexpr double kKeypointUncertaintyPx = 0.8;
};


/// \class GyroTracker
/// \brief Feature tracker using an interframe rotation matrix to predict the feature positions
///        while matching.
class GyroTracker : public FeatureTracker{
 public:
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(GyroTracker);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  /// \brief Construct the feature tracker.
  /// @param[in] input_camera The camera used in the tracker for projection/backprojection.
  explicit GyroTracker(const Camera& camera,
                       const size_t min_distance_to_image_border,
                       cv::Ptr<cv::DescriptorExtractor> extractor_ptr);
  virtual ~GyroTracker() {}

  /// \brief Track features between the current and the previous frames using a given interframe
  ///        rotation q_Ckp1_Ck to predict the feature positions.
  /// @param[in] q_Ckp1_Ck      Rotation matrix that describes the camera rotation between the
  ///                           two frames that are matched.
  /// @param[int] frame_k       The previous VisualFrame that needs to contain the keypoints and
  ///                           descriptor channels. Usually this is an output of the VisualPipeline.
  /// @param[out] frame_kp1     The current VisualFrame that needs to contain the keypoints and
  ///                           descriptor channels. Usually this is an output of the VisualPipeline.
  /// @param[out] matches_with_score_kp1_k  Vector of structs containing the found matches. Indices
  ///                                       correspond to the ordering of the keypoint/descriptor vector in the
  ///                                       respective frame channels.
  virtual void track(const Quaternion& q_Ckp1_Ck,
                     const VisualFrame& frame_k,
                     VisualFrame* frame_kp1,
                     MatchesWithScore* matches_with_score_kp1_k) override;

 private:
  struct TrackedMatch {
    TrackedMatch(const int index_k, const int index_km1)
      : correspondence{index_k, index_km1} {}
    std::array<int, 2u> correspondence;
  };

  enum class FeatureStatus {
    kDetected,
    kLkTracked
  };
  typedef std::vector<FeatureStatus> FrameFeatureStatus;
  typedef std::vector<size_t> FrameStatusTrackLength;

  virtual void LKTracking(
      const Eigen::Matrix2Xd& predicted_keypoint_positions_kp1,
      const std::vector<unsigned char>& prediction_success,
      const std::vector<int>& lk_candidate_indices_k,
      VisualFrame* frame_kp1,
      MatchesWithScore* matches_with_score_kp1_k);

  virtual void ComputeLKCandidates(
      const MatchesWithScore& matches_with_score_kp1_k,
      const FrameStatusTrackLength& status_track_length_k,
      const VisualFrame& frame_kp1,
      std::vector<int>* lk_candidate_indices_k) const;

  /// Compute matches from frame k and frame (k-1). They are called tracked
  /// matches since not all original matches get tracked (e.g. rejected by RANSAC).
  virtual void ComputeTrackedMatches(
      std::vector<TrackedMatch>* tracked_matches) const;

  virtual void ComputeUnmatchedIndicesOfFrameK(
      const MatchesWithScore& matches_with_score_kp1_k,
      std::vector<int>* unmatched_indices_k) const;

  virtual void ComputeStatusTrackLengthOfFrameK(
      const std::vector<TrackedMatch>& tracked_matches,
      FrameStatusTrackLength* status_track_length_k);

  virtual void InitializeFeatureStatusDeque();

  virtual void UpdateFeatureStatusDeque(
      const FrameFeatureStatus& frame_feature_status_kp1);

  virtual void UpdateFramePointerDeque(
      const VisualFrame* new_frame_k_ptr);

  template <typename Type>
  void EraseVectorElementsHelper(
          const std::unordered_set<size_t>& indices_to_erase,
          std::vector<Type>* vec) const;

  /// The camera model used in the tracker.
  const aslam::Camera& camera_;
  /// Minimum distance to image border is used to skip image points
  /// , predicted by the lk-tracker, that are too close to the image border.
  const size_t kMinDistanceToImageBorderPx;
  /// Remember if we have initialized already.
  bool initialized_;
  /// Descriptor extractor that is used on lk-tracked points.
  cv::Ptr<cv::DescriptorExtractor> extractor_;
  // Store pointers to frame k and (k-1) in that order.
  std::deque<const VisualFrame*> frames_k_km1_;
  /// Keep feature status for every index. For frames k and km1 in that order.
  std::deque<FrameFeatureStatus> feature_status_k_km1_;
  /// Keep status track length of frame (k-1) for every index.
  /// Status track length refers to the track length
  /// since the status of the feature has changed.
  FrameStatusTrackLength status_track_length_km1_;

  const GyroTrackerSettings settings;
};

// Erase elements of a vector based on a set of indices.
template <typename Type>
void GyroTracker::EraseVectorElementsHelper(
        const std::unordered_set<size_t>& indices_to_erase,
        std::vector<Type>* vec) const {
  CHECK_NOTNULL(vec);
    std::vector<bool> erase_index(vec->size(), false);
    for (const size_t i: indices_to_erase) {
        erase_index[i] = true;
    }
    std::vector<bool>::const_iterator it_to_erase = erase_index.begin();
    typename std::vector<Type>::iterator it_erase_from = std::remove_if(
        vec->begin(), vec->end(),
        [&it_to_erase](const Type& whatever) -> bool {
          return *it_to_erase++ == true;
        }
    );
    vec->erase(it_erase_from, vec->end());
    vec->shrink_to_fit();
}

} // namespace aslam

#endif  // ASLAM_GYRO_TRACKER_H_
