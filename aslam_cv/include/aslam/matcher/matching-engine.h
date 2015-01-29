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

  virtual bool match(MatchingProblem* problem, MatchesWithScore* matches) = 0;

  bool match_without_score(MatchingProblem* problem,
                           Matches* matches_0_1) {
    CHECK_NOTNULL(problem);
    CHECK_NOTNULL(matches_0_1);
    MatchesWithScore matches_with_score_0_1;
    const bool success = match(problem, &matches_with_score_0_1);
    convertMatch(&matches_with_score_0_1, matches_0_1);
    return success;
  }

  static void convertMatch(MatchesWithScore* matches_with_score_0_1,
                           Matches* matches_0_1) {
    for (const aslam::MatchWithScore& match : *matches_with_score_0_1) {
      CHECK_NE(match.getIndexApple(), -1) << "Apple keypoint index is -1.";
      CHECK_NE(match.getIndexBanana(), -1) << "Banana keypoint index is -1.";
      // N.b.: Matching from frame 0 to frame 1.
      matches_0_1->emplace_back(static_cast<size_t> (match.getIndexBanana()),
                                static_cast<size_t> (match.getIndexApple()));
    }
    CHECK_EQ(matches_with_score_0_1->size(), matches_0_1->size());
  }
};
}
#endif //ASLAM_CV_MATCHINGENGINE_H_
