#ifndef MATCHER_STEREO_MATCHER_H_
#define MATCHER_STEREO_MATCHER_H_

#include <algorithm>
#include <vector>

#include <Eigen/Core>
#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/cameras/undistorter-mapped.h>
#include <aslam/common/feature-descriptor-ref.h>
#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>

#include "aslam/matcher/match.h"

namespace aslam {

/// \class StereoMatcher
/// \brief Frame to frame matcher using the epipolar constraint to restrict
/// the search window.
/// The initial matcher attempts to match every keypoint of frame k to a
/// keypoint in frame (k+1). This is done by predicting the keypoint location
/// by using an interframe rotation matrix. Then a rectangular search window
/// around that location is searched for the best match greater than a
/// threshold. If the initial search was not successful, the search window is
/// increased once.
/// The initial matcher is allowed to discard a previous match if the new one
/// has a higher score. The discarded matches are called inferior matches and
/// a second matcher tries to match them. The second matcher only tries
/// to match a keypoint of frame k with the queried keypoints of frame (k+1)
/// of the initial matcher. Therefore, it does not compute distances between
/// descriptors anymore because the initial matcher has already done that.
/// The second matcher is executed several times because it is also allowed
/// to discard inferior matches of the current iteration.
/// The matches are exclusive.
class StereoMatcher {
 public:
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(StereoMatcher);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Constructs the StereoMatcher.
  /// @param[in]  stereo_pairs  The stereo pairs found in the current setup.
  // TODO(floriantschopp): update this!
  StereoMatcher(
      const size_t first_camera_idx, const size_t second_camera_idx,
      const aslam::NCamera::ConstPtr camera_rig,
      const Eigen::Matrix3d& fundamental_matrix,
      const Eigen::Matrix3d& rotation_C1_C0,
      const Eigen::Vector3d& translation_C1_C0,
      const std::shared_ptr<aslam::MappedUndistorter> first_mapped_undistorter,
      const std::shared_ptr<aslam::MappedUndistorter> second_mapped_undistorter,
      const aslam::VisualFrame::Ptr frame0,
      const aslam::VisualFrame::Ptr frame1,
      StereoMatchesWithScore* matches_frame0_frame1);
  virtual ~StereoMatcher(){};

  /// @param[in]  frame0        The first VisualFrame that needs to contain
  ///                           the keypoints and descriptor channels. Usually
  ///                           this is an output of the VisualPipeline.
  /// @param[in]  frame1        The second VisualFrame that needs to contain
  ///                           the keypoints and descriptor channels. Usually
  ///                           this is an output of the VisualPipeline.
  /// @param[out] matches_frame0_frame1 Vector of structs containing the found
  /// matches.
  ///                           Indices correspond to the ordering of the
  ///                           keypoint/descriptor vector in the respective
  ///                           frame channels.
  void match();

 private:
  struct KeypointData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeypointData(const Eigen::Vector2d& measurement, const int index)
        : measurement(measurement), channel_index(index) {}
    Eigen::Vector2d measurement;
    int channel_index;
  };

  typedef typename Aligned<std::vector, KeypointData>::const_iterator
      KeyPointIterator;
  typedef typename StereoMatchesWithScore::iterator MatchesIterator;

  struct MatchData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MatchData() = default;
    void addCandidate(
        const KeyPointIterator keypoint_iterator_frame1,
        const double matching_score) {
      CHECK_GT(matching_score, 0.0);
      CHECK_LE(matching_score, 1.0);
      keypoint_match_candidates_frame1.push_back(keypoint_iterator_frame1);
      match_candidate_matching_scores.push_back(matching_score);
    }
    // Iterators of keypoints of frame1 that were candidates for the match
    // together with their scores.
    std::vector<KeyPointIterator> keypoint_match_candidates_frame1;
    std::vector<double> match_candidate_matching_scores;
  };

  /// \brief Match a keypoint of frame0 with one of frame1 if possible.
  ///
  /// Initial matcher that tries to match a keypoint of frame0 with
  /// a keypoint of frame1 once. It is allowed to discard an
  /// already existing match.
  void matchKeypoint(const int idx_k);

  bool epipolarConstraint(
      const Eigen::Vector2d& keypoint_frame0,
      const Eigen::Vector2d& keypoint_frame1) const;

  /// \brief Try to match inferior matches without modifying initial matches.
  ///
  /// Second matcher that is only quering keypoints of frame1 that the
  /// initial matcher has queried before. Should be executed several times.
  /// Returns true if matches are still found.
  bool matchInferiorMatches(
      std::vector<bool>* is_inferior_keypoint_frame1_matched);

  int clamp(const int lower, const int upper, const int in) const;

  // The larger the matching score (which is smaller or equal to 1),
  // the higher the probability that a true match occurred.
  double computeMatchingScore(
      const int num_matching_bits,
      const unsigned int descriptor_size_bits) const;

  // Compute ratio test. Test is inspired by David Lowe's "ratio test"
  // for matching descriptors. Returns true if test is passed.
  bool ratioTest(
      const unsigned int descriptor_size_bits,
      const unsigned int distance_shortest,
      const unsigned int distance_second_shortest) const;

  // Triangulates found stereo pairs and calculates depth in each frame.
  // Returns true if sanaty check (depth > 0) passed.
  bool calculateDepth(aslam::StereoMatchWithScore* match); 

  const size_t first_camera_idx_;
  const size_t second_camera_idx_;
  const aslam::NCamera::ConstPtr camera_rig_;
  StereoMatchesWithScore* matches_frame0_frame1_;
  const Eigen::Matrix3d fundamental_matrix_;
  const Eigen::Matrix3d rotation_C1_C0_;
  const Eigen::Vector3d translation_C1_C0_;
  Eigen::Matrix3d camera_matrix_C0_inv_;
  Eigen::Matrix3d camera_matrix_C1_inv_;
  std::shared_ptr<MappedUndistorter> first_mapped_undistorter_;
  std::shared_ptr<MappedUndistorter> second_mapped_undistorter_;

  const aslam::VisualFrame::ConstPtr frame0_;
  const aslam::VisualFrame::ConstPtr frame1_;

  std::vector<common::FeatureDescriptorConstRef> descriptors_frame0_wrapped_;
  std::vector<common::FeatureDescriptorConstRef> descriptors_frame1_wrapped_;

  // Keypoints of frame1 sorted from small to large y coordinates.
  Aligned<std::vector, KeypointData> keypoints_frame1_sorted_by_y_;
  // corner_row_LUT[i] is the number of keypoints that has y position
  // lower than i in the image.
  std::vector<int> corner_row_LUT_;
  // Remember matched keypoints of frame1.
  std::vector<bool> is_keypoint_frame1_matched_;
  // Keep track of processed keypoints s.t. we don't process them again in the
  // large window. Set every element to false for each keypoint (of frame0)
  // iteration!
  std::vector<bool> iteration_processed_keypoints_frame1_;
  // Map from keypoint indices of frame1 to
  // the corresponding match iterator.
  std::unordered_map<int, MatchesIterator> frame1_idx_to_matches_iterator_map_;
  // The queried keypoints in frame1 and the corresponding
  // matching score are stored for each attempted match.
  // A map from the keypoint in frame0 to the corresponding
  // match data is created.
  std::unordered_map<int, MatchData> idx_frame0_to_attempted_match_data_map_;
  // Inferior matches are a subset of all attempted matches.
  // Remeber indices of keypoints in frame0 that are deemed inferior matches.
  std::vector<int> inferior_match_keypoint_idx_frame0_;

  const uint32_t kImageHeight;
  const int kNumPointsFrame0;
  const int kNumPointsFrame1;
  const double kEpipolarThreshold;
  const size_t kDescriptorSizeBytes;
  // Two descriptors could match if the number of matching bits normalized
  // with the descriptor length in bits is higher than this threshold.
  static constexpr float kMatchingThresholdBitsRatioRelaxed = 0.8f;
  // The more strict threshold is used for matching inferior matches.
  // It is more strict because there is no ratio test anymore.
  static constexpr float kMatchingThresholdBitsRatioStrict = 0.85f;
  // Two descriptors could match if they pass the Lowe ratio test.
  static constexpr float kLoweRatio = 0.8f;
  // Number of iterations to match inferior matches.
  static constexpr size_t kMaxNumInferiorIterations = 3u;
};

inline int StereoMatcher::clamp(
    const int lower, const int upper, const int in) const {
  return std::min<int>(std::max<int>(in, lower), upper);
}

inline double StereoMatcher::computeMatchingScore(
    const int num_matching_bits,
    const unsigned int descriptor_size_bits) const {
  return static_cast<double>(num_matching_bits) / descriptor_size_bits;
}

inline bool StereoMatcher::ratioTest(
    const unsigned int descriptor_size_bits,
    const unsigned int distance_closest,
    const unsigned int distance_second_closest) const {
  CHECK_LE(distance_closest, distance_second_closest);
  if (distance_second_closest > descriptor_size_bits) {
    // There has never been a second matching candidate.
    // Consequently, we cannot conclude with this test.
    return true;
  } else if (distance_second_closest == 0u) {
    // Unusual case of division by zero:
    // Let the ratio test be successful.
    return true;
  } else {
    return distance_closest / static_cast<float>(distance_second_closest) <
           kLoweRatio;
  }
}

}  // namespace aslam

#endif  // MATCHER_STEREO_MATCHER_H_
