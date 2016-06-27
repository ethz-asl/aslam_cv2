#ifndef ASLAM_CV_MATCHING_ENGINE_H_
#define ASLAM_CV_MATCHING_ENGINE_H_

#include <aslam/common/macros.h>

#include "aslam/matcher/match-helpers.h"

namespace aslam {

template<typename MATCHING_PROBLEM>
class MatchingEngine {
 public:
  typedef MATCHING_PROBLEM MatchingProblem;

  ASLAM_POINTER_TYPEDEFS(MatchingEngine);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngine);

  MatchingEngine() {};
  virtual ~MatchingEngine() {};

  virtual bool match(
      MatchingProblem* problem, typename MatchingProblem::MatchesWithScore* matches_A_B) = 0;

  virtual bool match(MatchingProblem* problem, typename MatchingProblem::Matches* matches_A_B) {
    CHECK_NOTNULL(problem);
    CHECK_NOTNULL(matches_A_B);
    typename MatchingProblem::MatchesWithScore matches_with_score_A_B;
    const bool success = match(problem, &matches_with_score_A_B);
    convertMatchesWithScoreToMatches<MatchingProblem>(matches_with_score_A_B, matches_A_B);
    return success;
  }
};

}  // namespace aslam
#endif // ASLAM_CV_MATCHING_ENGINE_H_
