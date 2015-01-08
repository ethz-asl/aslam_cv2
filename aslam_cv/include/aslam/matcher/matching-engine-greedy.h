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
  ASLAM_POINTER_TYPEDEFS(MatchingEngineGreedy);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineGreedy);

  MatchingEngineGreedy() {};
  virtual ~MatchingEngineGreedy() {};
  virtual bool match(MatchingProblem* problem, Matches* matches);
};

template<typename MatchingProblem>
bool MatchingEngineGreedy<MatchingProblem>::match(MatchingProblem* problem, Matches* matches) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(matches);
  matches->clear();
  bool status = problem->doSetup();
  size_t numA = problem->numApples();
  size_t numB = problem->numBananas();

  std::vector<typename MatchingProblem::Candidates> candidates(numB);

  int totalNumCandidates = 0;
  for (unsigned int b = 0; b < numB; ++b) {
    problem->getAppleCandidatesForBanana(b, &candidates[b]);
    totalNumCandidates += candidates[b].size();
  }

  matches->reserve(totalNumCandidates);
  for (unsigned int b = 0; b < numB; ++b) {
    // compute the score for each candidate and put in queue
    for (const typename MatchingProblem::Candidate& candidate_for_b : candidates[b]) {
      matches->emplace_back(candidate_for_b.index_apple, b, candidate_for_b.score);
    }
  }
  // reverse sort with reverse iterators
  std::sort(matches->rbegin(), matches->rend());

  // compress in place best unique match
  std::vector<unsigned char> assignedA(numA, false);

  auto match_out = matches->begin();
  for (auto match_in = matches->begin(); match_in != matches->end(); ++match_in) {
    int a = match_in->getIndexApple();

    if (!assignedA[a]) {
      assignedA[a] = true;
      *match_out++ = *match_in;
    }
  }

  // trim end of vector
  matches->erase(match_out, matches->end());

  return status;
}
}

#endif //ASLAM_CV_MATCHINGENGINE_GREEDY_H_
