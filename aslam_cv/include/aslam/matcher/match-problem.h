#ifndef ASLAM_CV_MATCH_VARIANT_H_
#define ASLAM_CV_MATCH_VARIANT_H_

/// \addtogroup Matcher
/// @{
///
/// @}

#include <vector>

#include <aslam/common/macros.h>

namespace aslam {

/// \class MatchProblem
///
/// \brief defines the specifics of a matching problem
///
/// The problem is assumed to have two lists (A and B) whose elements
/// can be referenced by a linear index. The problem defines the score
/// and scoring function between two elements of the lists, and a
/// method to get a short list of candiates from list A for elements
/// of list B.
///
template <typename SCORE_T>
class MatchProblem {
public:
  typedef SCORE_T score_t;
    
  struct Match {
    int correspondence[2];
    score_t score;
    int getIndexA() const { return correspondence[0]; }
    int getIndexB() const { return correspondence[1]; }
  };

  struct Candidate {
    int index;
    score_t score;
  };

  typedef std::vector<Match> Matches_t;
  typedef std::vector<Candidate> Candidates_t;

  virtual int getLengthA() =0;
  virtual int getLengthB() =0;

  /// get a short list of candidates in list a for index b
  /// return all indices of list a for n^2 matching; 
  /// or use something smarter like nabo to get nearest neighbors.
  /// can also be used to mask out invalid elements in the lists 
  /// or an invalid b, by returning and empty candidate list
  virtual int getCandidatesOfB(int b, Candidates_t *candidates)=0;
  virtual score_t computeScore(int a, int b)=0;

  /// gets called at the beginning of the matching problem; ie to setup kd-trees, lookup tables, whatever
  virtual void doSetup()=0;

  /// called at the end of the matching process to set the output 
  virtual void setBestMatches(const Matches_t &bestMatches)=0;

private:
  matches_t matches_;


};

}
#endif //ASLAM_CV_MATCH_VARIANT_H_
