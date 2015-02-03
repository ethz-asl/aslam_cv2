#ifndef ASLAM_CV_MATCHINGENGINE_H_
#define ASLAM_CV_MATCHINGENGINE_H_

#include <aslam/common/macros.h>
#include <aslam/matcher/match.h>
#include <aslam/matcher/matching-problem.h>

namespace aslam {

template<typename MATCHING_PROBLEM>
class MatchingEngine {
 public:
  typedef MATCHING_PROBLEM MatchingProblem;

  ASLAM_POINTER_TYPEDEFS(MatchingEngine);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngine);

  MatchingEngine() {};
  virtual ~MatchingEngine() {};

  virtual bool match(MatchingProblem* problem, MatchesWithScore* matches_A_B) = 0;

  bool match(MatchingProblem* problem, Matches* matches_A_B) {
    CHECK_NOTNULL(problem);
    CHECK_NOTNULL(matches_A_B);
    MatchesWithScore matches_with_score_A_B;
    const bool success = match(problem, &matches_with_score_A_B);
    convertMatches(matches_with_score_A_B, matches_A_B);
    return success;
  }
};
}
#endif //ASLAM_CV_MATCHINGENGINE_H_
