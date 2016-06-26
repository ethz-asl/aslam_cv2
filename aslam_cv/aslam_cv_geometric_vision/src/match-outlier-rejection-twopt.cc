#include <memory>
#include <vector>

#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/match.h>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/RotationOnlySacProblem.hpp>

namespace aslam {
namespace geometric_vision {

bool rejectOutlierFeatureMatchesTranslationSAC(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::Quaternion& q_Ckp1_Ck, const aslam::MatchesWithScore& matches_kp1_k,
    bool fix_random_seed, double ransac_threshold, size_t ransac_max_iterations,
    aslam::MatchesWithScore* inlier_matches_kp1_k,
    aslam::MatchesWithScore* outlier_matches_kp1_k) {
  CHECK_GT(ransac_threshold, 0.0);
  CHECK_GT(ransac_max_iterations, 0u);
  inlier_matches_kp1_k->clear();
  outlier_matches_kp1_k->clear();

  // Handle the case with too few matches to distinguish between out-/inliers.
  static constexpr size_t kMinKeypointCorrespondences = 6u;
  if (matches_kp1_k.size() < kMinKeypointCorrespondences) {
    VLOG(1) << "Too few matches to run RANSAC.";
    *outlier_matches_kp1_k =  matches_kp1_k;
    return false;
  }

  // Solve 2-pt RANSAC problem.
  opengv::bearingVectors_t bearing_vectors_kp1;
  opengv::bearingVectors_t bearing_vectors_k;

  aslam::Matches matches_without_score_kp1_k;
  aslam::convertMatches(matches_kp1_k, &matches_without_score_kp1_k);
  aslam::getBearingVectorsFromMatches(frame_kp1, frame_k, matches_without_score_kp1_k,
                                      &bearing_vectors_kp1, &bearing_vectors_k);

  using opengv::relative_pose::CentralRelativeAdapter;
  CentralRelativeAdapter adapter(bearing_vectors_kp1, bearing_vectors_k,
                                 q_Ckp1_Ck.getRotationMatrix());

  using opengv::sac_problems::relative_pose::TranslationOnlySacProblem;
  boost::shared_ptr<TranslationOnlySacProblem> twopt_problem(
      new TranslationOnlySacProblem(adapter, !fix_random_seed));

  opengv::sac::Ransac<TranslationOnlySacProblem> ransac;
  ransac.sac_model_ = twopt_problem;
  ransac.threshold_ = ransac_threshold;
  ransac.max_iterations_ = ransac_max_iterations;
  ransac.computeModel();

  if (ransac.inliers_.size() < kMinKeypointCorrespondences) {
    VLOG(1) << "Too few inliers to reliably classify outlier matches.";
    *outlier_matches_kp1_k =  matches_kp1_k;
    return false;
  }

  std::unordered_set<int> inlier_indicies(ransac.inliers_.begin(), ransac.inliers_.end());
  CHECK_EQ(ransac.inliers_.size(), inlier_indicies.size());

  // Remove the outliers from the matches list.
  int match_index = 0;
  for (const aslam::MatchWithScore& match : matches_kp1_k) {
    if (inlier_indicies.count(match_index)) {
      inlier_matches_kp1_k->emplace_back(match);
    } else {
      outlier_matches_kp1_k->emplace_back(match);
    }
    ++match_index;
  }
  CHECK_EQ(inlier_matches_kp1_k->size() + outlier_matches_kp1_k->size(), matches_kp1_k.size());
  return true;
}

bool rejectOutlierFeatureMatchesRelativePoseRotationSAC(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::Quaternion& q_Ckp1_Ck, const aslam::MatchesWithScore& matches_kp1_k,
    bool fix_random_seed, double ransac_threshold, size_t ransac_max_iterations,
    aslam::MatchesWithScore* inlier_matches_kp1_k,
    aslam::MatchesWithScore* outlier_matches_kp1_k) {
  CHECK_GT(ransac_threshold, 0.0);
  CHECK_GT(ransac_max_iterations, 0u);
  inlier_matches_kp1_k->clear();
  outlier_matches_kp1_k->clear();

  const size_t kNumMatches = matches_kp1_k.size();

  // Handle the case with too few matches to distinguish between out-/inliers.
  static constexpr size_t kMinKeypointCorrespondences = 6u;
  if (matches_kp1_k.size() < kMinKeypointCorrespondences) {
    VLOG(1) << "Too few matches to run RANSAC.";
    *outlier_matches_kp1_k =  matches_kp1_k;
    return false;
  }

  // Solve 2-pt RANSAC problem.
  opengv::bearingVectors_t bearing_vectors_kp1;
  opengv::bearingVectors_t bearing_vectors_k;

  aslam::Matches matches_without_score_kp1_k;
  aslam::convertMatches(matches_kp1_k, &matches_without_score_kp1_k);
  aslam::getBearingVectorsFromMatches(frame_kp1, frame_k, matches_without_score_kp1_k,
                                      &bearing_vectors_kp1, &bearing_vectors_k);
  using opengv::relative_pose::CentralRelativeAdapter;
  CentralRelativeAdapter adapter(bearing_vectors_kp1, bearing_vectors_k);
  /*
  CentralRelativeAdapter adapter(bearing_vectors_kp1, bearing_vectors_k,
                                 q_Ckp1_Ck.getRotationMatrix());
  */

  typedef opengv::sac_problems::relative_pose::RotationOnlySacProblem
      RotationOnlySacProblem;
  boost::shared_ptr<RotationOnlySacProblem> rotation_sac_problem(
      new RotationOnlySacProblem(adapter, !fix_random_seed));

  opengv::sac::Ransac<RotationOnlySacProblem> rotation_only_ransac;
  rotation_only_ransac.sac_model_ = rotation_sac_problem;
  rotation_only_ransac.threshold_ = ransac_threshold;
  //rotation_only_ransac.threshold_ = 9.0;
  rotation_only_ransac.max_iterations_ = ransac_max_iterations;
  //rotation_only_ransac.max_iterations_ = 50;
  rotation_only_ransac.computeModel();

  const size_t kNumRotationOnlyInliers = rotation_only_ransac.inliers_.size();
  const float rotation_only_inlier_ratio =
      static_cast<float>(kNumRotationOnlyInliers)/
      static_cast<float>(kNumMatches);

  typedef opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem
      CentralRelativePoseSacProblem;
  boost::shared_ptr<CentralRelativePoseSacProblem> relative_pose_sac_problem(
      new CentralRelativePoseSacProblem(
          adapter, CentralRelativePoseSacProblem::Algorithm::STEWENIUS,
          !fix_random_seed));
  opengv::sac::Ransac<CentralRelativePoseSacProblem> relative_pose_ransac;
  relative_pose_ransac.sac_model_ = relative_pose_sac_problem;
  //relative_pose_ransac.threshold_ = 9.0;
  relative_pose_ransac.threshold_ = ransac_threshold;
  //relative_pose_ransac.max_iterations_ = 50;
  relative_pose_ransac.max_iterations_ = ransac_max_iterations;
  relative_pose_ransac.computeModel();

  const size_t kNumRelativePoseInliers = relative_pose_ransac.inliers_.size();
  const float relative_pose_inlier_ratio =
      static_cast<float>(kNumRelativePoseInliers)/
      static_cast<float>(kNumMatches);

  bool use_rotation_ransac_result = false;
  bool use_relative_pose_ransac_result = false;
  std::vector<int>::const_iterator it_inliers_begin;
  std::vector<int>::const_iterator it_inliers_end;

  if (rotation_only_inlier_ratio > relative_pose_inlier_ratio ||
      rotation_only_inlier_ratio > 0.8) {
    if (kNumRotationOnlyInliers >= kMinKeypointCorrespondences) {
      use_rotation_ransac_result = true;
      it_inliers_begin = rotation_only_ransac.inliers_.begin();
      it_inliers_end = rotation_only_ransac.inliers_.end();
    }
  } else if (kNumRelativePoseInliers >= kMinKeypointCorrespondences) {
    use_relative_pose_ransac_result = true;
    it_inliers_begin = relative_pose_ransac.inliers_.begin();
    it_inliers_end = relative_pose_ransac.inliers_.end();
  }

  if (!use_rotation_ransac_result && ! use_relative_pose_ransac_result) {
    VLOG(1) << "Too few inliers to reliably classify outlier matches.";
    *outlier_matches_kp1_k =  matches_kp1_k;
    return false;
  } else {
    std::vector<bool> is_inlier_match(kNumMatches, false);
    for (std::vector<int>::const_iterator it = it_inliers_begin;
        it != it_inliers_end; ++it) {
      is_inlier_match.at(*it) = true;
      inlier_matches_kp1_k->emplace_back(matches_kp1_k.at(*it));
    }
    for (size_t index = 0u; index < kNumMatches; ++index) {
      if (!is_inlier_match[index]) {
        outlier_matches_kp1_k->emplace_back(matches_kp1_k.at(index));
      }
    }
    CHECK_EQ(inlier_matches_kp1_k->size() + outlier_matches_kp1_k->size(), matches_kp1_k.size());
    return true;
  }
}

}  // namespace gv
}  // namespace aslam
