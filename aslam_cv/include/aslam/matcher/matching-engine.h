#ifndef ASLAM_CV_MATCHINGENGINE_H_
#define ASLAM_CV_MATCHINGENGINE_H_

/// \addtogroup MatchingEngine
/// @{
///
/// @}


namespace aslam {

template <typename MATCHING_PROBLEM_T>
class MatchingEngine {

public:
  typedef MATCHING_PROBLEM_T Matching_Problem_t;
  typedef Matching_Problem_t::Score_t Score_t;

  virtual bool match(Matching_Problem &problem) = 0;

};



}
#endif //ASLAM_CV_MATCHINGENGINE_H_
