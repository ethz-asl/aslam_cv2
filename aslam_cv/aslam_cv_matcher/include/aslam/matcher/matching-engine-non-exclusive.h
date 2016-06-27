#ifndef ASLAM_CV_MATCHINGENGINE_NON_EXCLUSIVE_H_
#define ASLAM_CV_MATCHINGENGINE_NON_EXCLUSIVE_H_

#include <set>
#include <vector>

#include <aslam/common/macros.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

/// \brief Matching engine to simply return the best apple for each banana.
///        This explicitly does not deal with bananas matching to multiple apples and vice versa.
template<typename MatchingProblem>
class MatchingEngineNonExclusive : public MatchingEngine<MatchingProblem> {
 public:
  using MatchingEngine<MatchingProblem>::match;
  ASLAM_POINTER_TYPEDEFS(MatchingEngineNonExclusive);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineNonExclusive);

  MatchingEngineNonExclusive() {};
  virtual ~MatchingEngineNonExclusive() {};
  virtual bool match(MatchingProblem* problem,
                     typename MatchingProblem::MatchesWithScore* matches_A_B);
};

template<typename MatchingProblem>
bool MatchingEngineNonExclusive<MatchingProblem>::match(
    MatchingProblem* problem, typename MatchingProblem::MatchesWithScore* matches_A_B) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(matches_A_B);
  matches_A_B->clear();

  if (problem->doSetup()) {
    size_t num_bananas = problem->numBananas();

    typename MatchingProblem::CandidatesList candidates_for_bananas;
    problem->getCandidates(&candidates_for_bananas);
    CHECK_EQ(candidates_for_bananas.size(), num_bananas) << "The size of the candidates list does "
        << "not match the number of bananas of the problem. getCandidates(...) of the given "
        << "matching problem is supposed to return a vector of candidates for each banana and "
        << "hence the size of the returned vector must match the number of bananas.";
    for (size_t index_banana = 0u; index_banana < num_bananas; ++index_banana) {
      const typename MatchingProblem::Candidates& candidates =
          candidates_for_bananas[index_banana];

      typename MatchingProblem::Candidates::const_iterator best_candidate = candidates.begin();
      for (typename MatchingProblem::Candidates::const_iterator candidate_iterator =
          candidates.begin(); candidate_iterator != candidates.end(); ++candidate_iterator) {
        if (*candidate_iterator > *best_candidate) {
          best_candidate = candidate_iterator;
        }
      }

      if (best_candidate != candidates.end()) {
        matches_A_B->emplace_back(best_candidate->index_apple, index_banana, best_candidate->score);
      }
    }
    return true;
  } else {
    LOG(ERROR) << "Setting up the matching problem (.doSetup()) failed.";
    return false;
  }
}

}  // namespace aslam
#endif // ASLAM_CV_MATCHING_ENGINE_NON_EXCLUSIVE_H_
