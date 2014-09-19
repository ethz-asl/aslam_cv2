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
  typedef MATCHING_PROBLEM MatchingProblemT;
  typedef typename MatchingProblemT::ScoreT ScoreT;

  ASLAM_POINTER_TYPEDEFS(MatchingEngine);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngine);

  MatchingEngine() {};
  virtual ~MatchingEngine() {};

  virtual bool match(MatchingProblemT &problem) = 0;

};
}
#endif //ASLAM_CV_MATCHINGENGINE_H_
