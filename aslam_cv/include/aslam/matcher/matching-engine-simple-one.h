#ifndef ASLAM_CV_MATCHINGENGINE_SIMPLE_ONE_H_
#define ASLAM_CV_MATCHINGENGINE_SIMPLE_ONE_H_

#include <aslam/common/macros.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

template <typename MATCHING_PROBLEM>
class MatchingEngineSimpleOne2One : public MatchingEngine<MATCHING_PROBLEM> {

public:

  ASLAM_POINTER_TYPEDEFS(MatchingEngineSimpleOne2One);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineSimpleOne2One);

  MatchingEngineSimpleOne2One();
  virtual ~MatchingEngineSimpleOne2One();
  virtual bool match(MATCHING_PROBLEM &problem);

};

}

#endif //ASLAM_CV_MATCHINGENGINE_SIMPLE_ONE_H_
