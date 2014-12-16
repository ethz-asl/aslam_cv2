#ifndef ASLAM_CV_MATCHING_PROBLEM_H_
#define ASLAM_CV_MATCHING_PROBLEM_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <vector>

#include <aslam/common/macros.h>
#include <glog/logging.h>

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
template<CandidateScore CandiateScoreType>
class MatchingProblem {
public:
  struct Candidate {
    int index_apple;
    double score; /// a preliminary score that can be used for
                   /// sorting, rough thresholding; but actual match
                   /// score will get recomputed.
    Candidate(int _index_apple, const double& _score) : index_apple(_index_apple), score(_score) {}
  };

  typedef std::vector<Candidate> Candidates;

  ASLAM_POINTER_TYPEDEFS(MatchingProblem);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblem);

  MatchingProblem() {};
  virtual ~MatchingProblem() {};

  virtual size_t numApples() const = 0;
  virtual size_t numBananas() const = 0;

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
  virtual void getAppleCandidatesForBanana(int /*b*/, Candidates *candidates) {
    CHECK_NOTNULL(candidates);
    candidates->clear();
    candidates->reserve(numApples());

    // just returns all apples with no score
    for (unsigned int i = 0; i < numApples(); ++i) {
      candidates->emplace_back(i, 0);
    }
  };

  /// \brief compute the match score between items referenced by a and b.
  /// Note: this can be called multiple times from different threads.
  /// Warning: these are scores and *not* distances, higher values are better
  virtual double computeScore(int a, int b) {
    LOG(FATAL) << "Not implemented! If this function is called, it means that the candidate score "
        "is defined as not final in the matching problem derived class. In this case, this virtual "
        "function (computeScore(a, b) has to be implemented in the derived class.";
        return 0.0; }

  /// Gets called at the beginning of the matching problem; ie to setup kd-trees, lookup tables, whatever...
  virtual bool doSetup() = 0;

  const CandidateScore kCandidateSoreType = CandiateScoreType;
};
}
#endif //ASLAM_CV_MATCHING_PROBLEM_H_
