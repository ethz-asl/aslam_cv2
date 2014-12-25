#ifndef ASLAM_CV_MATCHING_PROBLEM_H_
#define ASLAM_CV_MATCHING_PROBLEM_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <set>
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
class MatchingProblem {
public:
  struct Candidate {
    int index_apple;
    int index_banana;
    double score;
    /// \brief The priority field allows categorizing candidates. Certain matching engines might
    ///        treat candidates differently according to their priority.
    ///        The priority outrules the score in the candidate comparison.
    int priority;

    Candidate() : index_apple(-1), index_banana(-1), score(0.0), priority(-1) {}
    Candidate(int _index_apple, int _index_banana, double _score, int _priority) :
      index_apple(_index_apple), index_banana(_index_banana), score(_score), priority(_priority) {}

    bool operator<(const Candidate& other) const {
      return (this->priority < other.priority) ||
          ((this->priority == other.priority) && (this->score < other.score));
    }
    bool operator>(const Candidate& other) const {
      return (this->priority > other.priority) ||
          ((this->priority == other.priority) && (this->score > other.score));
    }

    bool operator==(const Candidate& other) const {
      return (this->index_apple == other.index_apple) &&
             (this->score == other.score) &&
             (this->priority == other.priority);
    }
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
  virtual void getAppleCandidatesForBanana(int /*b*/, Candidates* candidates) = 0;

  /// Gets called at the beginning of the matching problem; ie to setup kd-trees, lookup tables, whatever...
  virtual bool doSetup() = 0;
};
}
#endif //ASLAM_CV_MATCHING_PROBLEM_H_
