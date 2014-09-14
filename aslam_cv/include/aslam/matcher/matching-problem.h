#ifndef ASLAM_CV_MATCHING_PROBLEM_H_
#define ASLAM_CV_MATCHING_PROBLEM_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <vector>

#include <aslam/common/macros.h>
#include "match.h"
namespace aslam {

/// \class MatchingProblem
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
class MatchingProblem {
public:
  typedef SCORE_T Score_t;
    
  typedef Match<Score_t> Match;

  struct Candidate {
    int index;
    Score_t score; /// a preliminary score that can be used for
                   /// sorting, rough thresholding; but actuall match
                   /// score will get recomputed.
  };

  typedef std::vector<Match> Matches_t;
  typedef std::vector<Candidate> Candidates_t;

  virtual int getLengthA() = 0;
  virtual int getLengthB() = 0;

  /// Get a short list of candidates in list a for index b
  ///
  /// Return all indices of list a for n^2 matching; or use something
  /// smarter like nabo to get nearest neighbors.  Can also be used to
  /// mask out invalid elements in the lists, or an invalid b, by
  /// returning and empty candidate list.  
  ///
  /// The score for each candidate is a rough score that can be used
  /// for sorting, pre-filtering, and will be explicitly recomputed
  /// using the computeScore function.
  ///
  /// \param[in] b The index of b queried for candidates.
  /// \param[out] candidates Candidates from the A-list that could potentially match this element of B.
  virtual int getCandidatesOfB(int b, Candidates_t *candidates) = 0;

  /// \brief compute the match score between items referenced by a and b.
  /// Note: this will be called multilple times from different threads.
  virtual Score_t computeScore(int a, int b) = 0;

  /// Gets called at the beginning of the matching problem; ie to setup kd-trees, lookup tables, whatever...
  virtual bool doSetup() = 0;

  /// Called at the end of the matching process to set the output. 
  virtual void setBestMatches(const Matches_t &bestMatches) = 0;

};
}
#endif //ASLAM_CV_MATCHING_PROBLEM_H_
