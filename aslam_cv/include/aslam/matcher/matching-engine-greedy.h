#ifndef ASLAM_CV_MATCHINGENGINE_GREEDY_H_
#define ASLAM_CV_MATCHINGENGINE_GREEDY_H_

#include <vector>

#include <aslam/common/macros.h>
#include <aslam/matcher/match.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

template<typename MatchingProblem>
class MatchingEngineGreedy : public MatchingEngine<MatchingProblem> {
 public:
  using MatchingEngine<MatchingProblem>::match;
  ASLAM_POINTER_TYPEDEFS(MatchingEngineGreedy);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineGreedy);

  MatchingEngineGreedy() {};
  virtual ~MatchingEngineGreedy() {};
  virtual bool match(MatchingProblem* problem, MatchesWithScore* matches_A_B);
};

template<typename MatchingProblem>
bool MatchingEngineGreedy<MatchingProblem>::match(MatchingProblem* problem,
						  MatchesWithScore* matches_A_B) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(matches_A_B);
  matches_A_B->clear();
  bool status = problem->doSetup();
  size_t numA = problem->numApples();
  size_t numB = problem->numBananas();

  std::vector<typename MatchingProblem::Candidates> candidates(numB);

  int totalNumCandidates = 0;
  for (unsigned int b = 0; b < numB; ++b) {
    problem->getAppleCandidatesForBanana(b, &candidates[b]);
    totalNumCandidates += candidates[b].size();
  }

  matches_A_B->reserve(totalNumCandidates);
  for (unsigned int b = 0; b < numB; ++b) {
    // compute the score for each candidate and put in queue
    for (const typename MatchingProblem::Candidate& candidate_for_b : candidates[b]) {
      matches_A_B->emplace_back(candidate_for_b.index_apple, b, candidate_for_b.score);
    }
  }
  // reverse sort with reverse iterators
  std::sort(matches_A_B->rbegin(), matches_A_B->rend());

  // compress in place best unique match
  std::vector<unsigned char> assignedA(numA, false);

  auto match_out = matches_A_B->begin();
  for (auto match_in = matches_A_B->begin(); match_in != matches_A_B->end(); ++match_in) {
    int a = match_in->getIndexApple();

    if (!assignedA[a]) {
      assignedA[a] = true;
      *match_out++ = *match_in;
    }
  }

  // trim end of vector
  matches_A_B->erase(match_out, matches_A_B->end());

  return status;
}

}

#endif //ASLAM_CV_MATCHINGENGINE_GREEDY_H_
