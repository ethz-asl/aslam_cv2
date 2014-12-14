#ifndef ASLAM_CV_MATCHINGENGINE_NON_EXCLUSIVE_H_
#define ASLAM_CV_MATCHINGENGINE_NON_EXCLUSIVE_H_

#include <glog/logging.h>
#include <vector>

#include <aslam/common/macros.h>
#include <aslam/matcher/match.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

/// \brief Matching engine to simply return all candidate matches from the given matching problem.
///        This explicitely does not deal with bananas matching to multiple apples and vice versa.
template<typename MatchingProblem>
class MatchingEngineNonExclusive : public MatchingEngine<MatchingProblem> {
 public:
  ASLAM_POINTER_TYPEDEFS(MatchingEngineNonExclusive);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineNonExclusive);

  MatchingEngineNonExclusive() {};
  virtual ~MatchingEngineNonExclusive() {};
  virtual bool match(MatchingProblem* problem, typename aslam::Matches* matches);
};

template<typename MatchingProblem>
bool MatchingEngineNonExclusive<MatchingProblem>::match(MatchingProblem* problem,
                                                        aslam::Matches* matches) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(matches);
  matches->clear();

  if (problem->doSetup()) {
    size_t num_bananas = problem->numBananas();

    for (size_t banana_index = 0; banana_index < num_bananas; ++banana_index) {
      typename MatchingProblem::Candidates candidates;
      problem->getAppleCandidatesForBanana(banana_index, &candidates);
      for (auto it = candidates.begin(); it != candidates.end(); ++it) {
        matches->emplace_back(it->index_apple, banana_index, it->score);
      }
    }
    LOG(INFO) << "Matched " << matches->size() << " keypoints.";
    return true;
  } else {
    LOG(ERROR) << "Setting up the maching problem (.doSetup()) failed.";
    return false;
  }
}

}  // namespace aslam

#endif //ASLAM_CV_MATCHINGENGINE_NON_EXCLUSIVE_H_
