#include <memory>
#include <vector>

#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-nframe.h>
#include <aslam/matcher/match.h>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>

namespace aslam {
namespace gv {
bool rejectOutlierKeypointMatchesTwopt(const aslam::VisualFrame& frame_kp1,
                                         const aslam::VisualFrame& frame_k,
                                         const aslam::Quaternion& q_Ckp1_Ck,
                                         const aslam::MatchesWithScore& matches_kp1_k,
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
      new TranslationOnlySacProblem(adapter));

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

}  // namespace gv
}  // namespace aslam
