#ifndef ASLAM_CV_MATCHINGENGINE_H_
#define ASLAM_CV_MATCHINGENGINE_H_

#include <aslam/common/macros.h>

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

template <typename MATCHING_PROBLEM>
class MatchingEngine {

public:
  typedef MATCHING_PROBLEM Matching_ProblemT;
  typedef Matching_ProblemT::ScoreT ScoreT;

  ASLAM_POINTER_TYPEDEFS(MatchingEngine);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngine);

  MatchingEngine();
  virtual ~MatchingEngine();

  virtual bool match(Matching_ProblemT &problem) = 0;

};
}
#endif //ASLAM_CV_MATCHINGENGINE_H_
