#ifndef ASLAM_CV_MATCHINGENGINE_GREEDY_H_
#define ASLAM_CV_MATCHINGENGINE_GREEDY_H_

#include <vector>

#include <aslam/common/macros.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

template <typename MatchingProblem>
class MatchingEngineGreedy : public MatchingEngine<MatchingProblem> {

public:

  ASLAM_POINTER_TYPEDEFS(MatchingEngineGreedy);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineGreedy);

  MatchingEngineGreedy() {};
  virtual ~MatchingEngineGreedy() {};
  virtual bool match(MatchingProblem &problem);

};

template <typename MatchingProblem>
bool MatchingEngineGreedy<MatchingProblem>::match(MatchingProblem &problem)
{

  bool status = problem.doSetup();

  int numA = problem.getLengthApples();
  int numB = problem.getLengthBananas();

  typename MatchingProblem::MatchesT matches;

  std::vector<typename MatchingProblem::CandidatesT> candidates(numB);

  int a,b;

  int totalNumCandidates=0;
  for (b = 0; b < numB; ++b) {
    problem.getAppleCandidatesOfBanana(b,&candidates[b]);
    totalNumCandidates += candidates[b].size();
  }

  matches.reserve(totalNumCandidates);
  for (b = 0; b < numB; ++b) {
    // compute the score for each candidate and put in queue
    for (int c = 0; c < candidates[b].size(); ++c) {
      a = candidates[b][c].index;

      typename MatchingProblem::ScoreT score = problem.computeScore(a,b);
      matches.emplace_back(a, b, score);
    }
  }
  // reverse sort with reverse iterators
  std::sort(matches.rbegin(), matches.rend());

  // compress in place best unique match
  std::vector<bool> assignedA(numA, false);
  auto match_out = matches.begin();
  for (auto match_in = matches.begin(); match_in != matches.end(); ++match_in) {
    a = match_in->getIndexA();
    b = match_in->getIndexB();
    if (!assignedA[a]) {
      assignedA[a] = true;
      *match_out++ = *match_in;
    }
  }

  // trim end of vector
  matches.erase(match_out, matches.end());

  problem.setBestMatches(matches);

  return status;
}


}

#endif //ASLAM_CV_MATCHINGENGINE_GREEDY_H_
