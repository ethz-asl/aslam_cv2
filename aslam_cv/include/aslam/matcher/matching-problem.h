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
/// The problem is assumed to have two lists (Apples and Bananas) whose elements
/// can be referenced by a linear index. The problem defines the score
/// and scoring function between two elements of the lists, and a
/// method to get a short list of candidates from list Apples for elements
/// of list Bananas.
/// 
/// The match is not necessarily symmetric. For example, Apples can
/// represent a reference and Bananas queries.
///
template <typename SCORE>
class MatchingProblem {
public:
  typedef SCORE ScoreT;
    
  typedef Match<ScoreT> MatchT;

  struct Candidate {
    int index;
    ScoreT score; /// a preliminary score that can be used for
                   /// sorting, rough thresholding; but actual match
                   /// score will get recomputed.
    Candidate(int _index, const ScoreT& _score) : index(_index), score(_score) {}
  };

  typedef std::vector<MatchT> MatchesT;
  typedef std::vector<Candidate> CandidatesT;

  ASLAM_POINTER_TYPEDEFS(MatchingProblem);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblem);

  MatchingProblem() {};
  virtual ~MatchingProblem() {};

  virtual int getLengthApples() const = 0;
  virtual int getLengthBananas() const = 0;

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
  /// \param[out] candidates Candidates from the Apples-list that could potentially match this element of Bananas.
  virtual void getAppleCandidatesOfBanana(int /*b*/, CandidatesT *candidates) {
    CHECK_NOTNULL(candidates);
    candidates->clear();
    candidates->reserve(getLengthApples());

    // just returns all apples with no score
    for (int i = 0; i < getLengthApples(); ++i) {
      candidates->emplace_back(i, 0);
    }
  };

  /// \brief compute the match score between items referenced by a and b.
  /// Note: this can be called multiple times from different threads.
  /// Warning: these are scores and *not* distances, higher values are better
  virtual ScoreT computeScore(int a, int b) = 0;

  /// Gets called at the beginning of the matching problem; ie to setup kd-trees, lookup tables, whatever...
  virtual bool doSetup() = 0;

  /// Called at the end of the matching process to set the output. 
  virtual void setBestMatches(const MatchesT &bestMatches) = 0;

};
}
#endif //ASLAM_CV_MATCHING_PROBLEM_H_
