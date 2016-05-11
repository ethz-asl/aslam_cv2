#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

#include <vector>

#include <aslam/common/pose-types.h>
#include <aslam/common/stl-helpers.h>
#include <Eigen/Core>
#include <opencv2/features2d/features2d.hpp>

namespace aslam {
class VisualFrame;
class VisualNFrame;

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
struct MatchWithScore;
typedef std::vector<MatchWithScore> MatchesWithScore;
typedef std::pair<size_t, size_t> Match;
typedef std::vector<Match> Matches;
typedef std::vector<cv::DMatch> OpenCvMatches;

struct MatchWithScore {
  friend class SimpleMatchProblem;
  friend class MatchingProblemFrameToFrame;
  friend class MatchingProblemLandmarksToFrame;
  friend class SimpleTrackManager;
  friend class UniformTrackManager;
  template<typename MatchingProblem> friend class MatchingEngineGreedy  ;
  friend void convertMatches(const MatchesWithScore&, Matches*);
  friend void convertMatches(const MatchesWithScore&, OpenCvMatches*);
  /// \brief Initialize to an invalid match.
  MatchWithScore()
      : correspondence {-1, -1}, score(0.0) {}

  /// \brief Initialize with correspondences and a score.
  MatchWithScore(int index_apple, int index_banana, double _score)
      : correspondence {index_apple, index_banana}, score(_score) {}

  void setIndexApple(int index_apple) {
    correspondence[0] = index_apple;
  }

  void setIndexBanana(int index_banana) {
      correspondence[1] = index_banana;
  }

  void setScore(double _score) {
    score = _score;
  }

  /// \brief Get the score given to the match.
  double getScore() const {
    return score;
  }

  bool operator<(const MatchWithScore &other) const {
    return this->score < other.score;
  }
  bool operator>(const MatchWithScore &other) const {
    return this->score > other.score;
  }

  bool operator==(const MatchWithScore& other) const {
    return (this->correspondence[0] == other.correspondence[0]) &&
           (this->correspondence[1] == other.correspondence[1]) &&
           (this->score == other.score);
  }

 private:
  /// \brief Get the index into list A.
  int getIndexApple() const {
    return correspondence[0];
  }

  /// \brief Get the index into list B.
  int getIndexBanana() const {
    return correspondence[1];
  }

  int correspondence[2];
  double score;
};

/// Convert MatchesWithScore to Matches.
void convertMatches(const MatchesWithScore& matches_with_score_A_B, Matches* matches_A_B);

/// Convert MatchesWithScore to cv::DMatches.
void convertMatches(const MatchesWithScore& matches_with_score_A_B, OpenCvMatches* matches_A_B);

/// Get number of matches for a rig match list. (outer vector = cameras, inner vector = match list)
template<typename MatchType>
inline size_t countRigMatches(const std::vector<std::vector<MatchType>>& rig_matches) {
  size_t num_matches = 0;
  for (const std::vector<MatchType>& camera_matches : rig_matches) {
    num_matches += camera_matches.size();
  }
  return num_matches;
}

/// Select and return N random matches for each camera in the rig.
template<typename MatchesType>
std::vector<MatchesType> pickNRandomRigMatches(size_t n_per_camera,
                                               const std::vector<MatchesType>& rig_matches) {
  CHECK_GT(n_per_camera, 0u);
  size_t num_cameras = rig_matches.size();
  std::vector<MatchesType> subsampled_rig_matches(num_cameras);

  for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
    const MatchesType& camera_matches = rig_matches[cam_idx];
    if (camera_matches.size() <= n_per_camera) {
      subsampled_rig_matches[cam_idx] = camera_matches;
    } else {
      common::drawNRandomElements(n_per_camera, camera_matches, &subsampled_rig_matches[cam_idx]);
    }
  }
  CHECK_EQ(rig_matches.size(), subsampled_rig_matches.size());
  return subsampled_rig_matches;
}

/// Get the matches based on the track id channels for one VisualFrame.
size_t extractMatchesFromTrackIdChannel(const aslam::VisualFrame& frame_kp1,
                                        const aslam::VisualFrame& frame_k,
                                        aslam::Matches* matches_kp1_kp);

/// Get the matches based on the track id channels for one VisualNFrame.
size_t extractMatchesFromTrackIdChannels(const aslam::VisualNFrame& nframe_kp1,
                                         const aslam::VisualNFrame& nframe_k,
                                         std::vector<aslam::Matches>* rig_matches_kp1_kp);

/// Get the median pixel disparity for all matches.
double getMatchPixelDisparityMedian(const aslam::VisualNFrame& nframe_kp1,
                                    const aslam::VisualNFrame& nframe_k,
                                    const std::vector<aslam::Matches>& matches_kp1_kp);

/// Get the median pixel disparity for all matches, taking into account the relative
/// orientation of the frames.
double getUnrotatedMatchPixelDisparityMedian(const aslam::VisualNFrame& nframe_kp1,
                                             const aslam::VisualNFrame& nframe_k,
                                             const std::vector<aslam::Matches>& matches_kp1_kp,
                                             const aslam::Quaternion& q_kp1_k);

/// Return the normalized bearing vectors for a list of single camera matches.
void getBearingVectorsFromMatches(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::Matches& matches_kp1_k,
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_kp1,
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_k);

}  // namespace aslam

#endif // ASLAM_MATCH_H_
