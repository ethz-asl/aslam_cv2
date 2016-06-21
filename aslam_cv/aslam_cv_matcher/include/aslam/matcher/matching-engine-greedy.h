#ifndef ASLAM_CV_MATCHING_ENGINE_GREEDY_H_
#define ASLAM_CV_MATCHING_ENGINE_GREEDY_H_

#include <vector>

#include <aslam/common/macros.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

template<typename MatchingProblem>
class MatchingEngineGreedy : public MatchingEngine<MatchingProblem> {
 public:
  ASLAM_POINTER_TYPEDEFS(MatchingEngineGreedy);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineGreedy);

  MatchingEngineGreedy() {};
  virtual ~MatchingEngineGreedy() {};
  virtual bool match(MatchingProblem* problem,
                     typename MatchingProblem::MatchesWithScore* matches_A_B);
};

template<typename MatchingProblem>
bool MatchingEngineGreedy<MatchingProblem>::match(
    MatchingProblem* problem, typename MatchingProblem::MatchesWithScore* matches_A_B) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(matches_A_B);
  matches_A_B->clear();
  if (problem->doSetup()) {
    const size_t num_apples = problem->numApples();
    const size_t num_bananas = problem->numBananas();

    typename MatchingProblem::CandidatesList candidates;
    problem->getCandidates(&candidates);
    CHECK_EQ(candidates.size(), num_bananas) << "The size of the candidates list does not "
        << "match the number of bananas of the problem. getCandidates(...) of the given matching "
        << "problem is supposed to return a vector of candidates for each banana and hence the "
        << "size of the returned vector must match the number of bananas.";

    size_t total_num_candidates = 0u;
    for (const typename MatchingProblem::Candidates& candidates_for_banana : candidates) {
      total_num_candidates += candidates_for_banana.size();
    }

    matches_A_B->reserve(total_num_candidates);
    for (size_t banana_idx = 0u; banana_idx < num_bananas; ++banana_idx) {
      // compute the score for each candidate and put in queue
      for (const typename MatchingProblem::Candidate& candidate_for_banana :
          candidates[banana_idx]) {
        matches_A_B->emplace_back(
            candidate_for_banana.index_apple, banana_idx, candidate_for_banana.score);
      }
    }
    // Reverse sort with reverse iterators.
    std::sort(matches_A_B->rbegin(), matches_A_B->rend());

    // Compress the best unique match in place.
    std::vector<unsigned char> is_apple_assigned(num_apples, false);

    typename MatchingProblem::MatchesWithScore::iterator output_match_iterator =
        matches_A_B->begin();
    for (const typename MatchingProblem::MatchWithScore& match : *matches_A_B) {
      const int apple_index = match.getIndexApple();

      if (!is_apple_assigned[apple_index]) {
        is_apple_assigned[apple_index] = true;
        *output_match_iterator++ = match;
      }
    }

    // Trim the end of the vector.
    matches_A_B->erase(output_match_iterator, matches_A_B->end());
  } else {
    LOG(WARNING) << "problem->doSetup() failed.";
    return false;
  }

  return true;
}

}  // namespace aslam

#endif // ASLAM_CV_MATCHING_MATCHING_ENGINE_GREEDY_H_
