#ifndef ASLAM_CV_MATCHINGENGINE_GREEDY_H_
#define ASLAM_CV_MATCHINGENGINE_GREEDY_H_

#include <aslam/common/macros.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

template <typename MATCHING_PROBLEM>
class MatchingEngineGreedy : public MatchingEngine<MATCHING_PROBLEM> {

public:

  ASLAM_POINTER_TYPEDEFS(MatchingEngineGreedy);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineGreedy);

  MatchingEngineGreedy();
  virtual ~MatchingEngineGreedy();
  virtual bool match(MATCHING_PROBLEM &problem);

};

}

#endif //ASLAM_CV_MATCHINGENGINE_GREEDY_H_
