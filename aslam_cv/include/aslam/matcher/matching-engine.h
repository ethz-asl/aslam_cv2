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

  virtual bool match(MatchingProblem* problem, Matches* matches) = 0;
  virtual bool match(MatchingProblem* problem, std::vector<std::pair<size_t,size_t> >& matches) = 0;
};
}
#endif //ASLAM_CV_MATCHINGENGINE_H_
