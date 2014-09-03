#ifndef ASLAM_CV_MATCHER_H_
#define ASLAM_CV_MATCHER_H_

/// \addtogroup Matcher
/// @{
///
/// @}


namespace aslam {

template <typename MATCH_PROBLEM_T>
class Matcher {

public:
  typedef MATCH_PROBLEM_T Match_Problem_t;
  typedef Match_Problem_t::Score_t Score_t;

  virtual match(Match_Problem &problem)=0;

};



}
#endif //ASLAM_CV_MATCHER_H_
