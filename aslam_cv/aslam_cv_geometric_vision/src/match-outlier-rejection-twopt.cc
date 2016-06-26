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

bool rejectOutlierFeatureMatchesTranslationRotationSAC(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::Quaternion& q_Ckp1_Ck,
    const aslam::MatchesWithScore& matches_kp1_k, bool fix_random_seed,
    double ransac_threshold, size_t ransac_max_iterations,
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

  opengv::bearingVectors_t bearing_vectors_kp1;
  opengv::bearingVectors_t bearing_vectors_k;

  aslam::Matches matches_without_score_kp1_k;
  aslam::convertMatches(matches_kp1_k, &matches_without_score_kp1_k);
  aslam::getBearingVectorsFromMatches(frame_kp1, frame_k, matches_without_score_kp1_k,
                                      &bearing_vectors_kp1, &bearing_vectors_k);
  using opengv::relative_pose::CentralRelativeAdapter;
  CentralRelativeAdapter adapter(bearing_vectors_kp1, bearing_vectors_k,
                                 q_Ckp1_Ck.getRotationMatrix());

  typedef opengv::sac_problems::relative_pose::RotationOnlySacProblem
      RotationOnlySacProblem;
  boost::shared_ptr<RotationOnlySacProblem> rotation_sac_problem(
      new RotationOnlySacProblem(adapter, !fix_random_seed));

  opengv::sac::Ransac<RotationOnlySacProblem> rotation_ransac;
  rotation_ransac.sac_model_ = rotation_sac_problem;
  rotation_ransac.threshold_ = ransac_threshold;
  rotation_ransac.max_iterations_ = ransac_max_iterations;
  rotation_ransac.computeModel();

  typedef opengv::sac_problems::relative_pose::TranslationOnlySacProblem
      TranslationOnlySacProblem;
  boost::shared_ptr<TranslationOnlySacProblem> translation_sac_problem(
      new TranslationOnlySacProblem(
          adapter, !fix_random_seed));
  opengv::sac::Ransac<TranslationOnlySacProblem> translation_ransac;
  translation_ransac.sac_model_ = translation_sac_problem;
  translation_ransac.threshold_ = ransac_threshold;
  translation_ransac.max_iterations_ = ransac_max_iterations;
  translation_ransac.computeModel();

  // Take the union of both inlier sets as final inlier set.
  std::unordered_set<int> inlier_indices(
      rotation_ransac.inliers_.begin(), rotation_ransac.inliers_.end());
  inlier_indices.insert(translation_ransac.inliers_.begin(), translation_ransac.inliers_.end());

  // Remove the outliers from the matches list.
  int match_index = 0;
  for (const aslam::MatchWithScore& match : matches_kp1_k) {
    if (inlier_indices.count(match_index)) {
      inlier_matches_kp1_k->emplace_back(match);
    } else {
      outlier_matches_kp1_k->emplace_back(match);
    }
    ++match_index;
  }
  CHECK_EQ(inlier_matches_kp1_k->size() + outlier_matches_kp1_k->size(), matches_kp1_k.size());
  return true;
}

}  // namespace gv
}  // namespace aslam
