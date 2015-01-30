#ifndef ASLAM_CV_MATCHINGENGINE_H_
#define ASLAM_CV_MATCHINGENGINE_H_

#include <aslam/common/macros.h>
#include <aslam/matcher/match.h>
#include <aslam/matcher/matching-problem.h>
/// \addtogroup Matching
/// @{
///
/// @}

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
    convertMatch(&matches_with_score_A_B, matches_A_B);
    return success;
  }

  static void convertMatch(MatchesWithScore* matches_with_score_A_B,
                           Matches* matches_A_B) {
    CHECK_NOTNULL(matches_with_score_A_B);
    CHECK_NOTNULL(matches_A_B);
    for (const aslam::MatchWithScore& match : *matches_with_score_A_B) {
      CHECK_GE(match.getIndexApple(), 0) << "Apple keypoint index is negative.";
      CHECK_GE(match.getIndexBanana(), 0) << "Banana keypoint index is negative.";
      matches_A_B->emplace_back(static_cast<size_t> (match.getIndexApple()),
                                static_cast<size_t> (match.getIndexBanana()));
    }
    CHECK_EQ(matches_with_score_A_B->size(), matches_A_B->size());
  }
};
}
#endif //ASLAM_CV_MATCHINGENGINE_H_
