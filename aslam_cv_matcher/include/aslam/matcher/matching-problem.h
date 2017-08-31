#ifndef ASLAM_CV_MATCHING_PROBLEM_H_
#define ASLAM_CV_MATCHING_PROBLEM_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <vector>

#include <aslam/common/macros.h>
#include <aslam/common/memory.h>
#include <glog/logging.h>

namespace aslam {

/// \class MatchingProblem
///
/// \brief defines the specifics of a matching problem
///
/// IMPORTANT NOTE: If you create a new matching problem inheriting from this class, make sure
/// not to forget to add the
/// ASLAM_ADD_MATCH_TYPEDEFS_WITH_ALIASES(
///    NewMatchingProblemClassName, alias_for_getAppleIndex, alias_for_getBananaIndex)
/// macro in the public class body! Otherwise, no matches type will be available for the
/// new problem and compilation will fail.
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

  typedef Aligned<std::vector, Candidate> Candidates;
  typedef Aligned<std::vector, Candidates> CandidatesList;

  ASLAM_POINTER_TYPEDEFS(MatchingProblem);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblem);

  MatchingProblem() = default;
  virtual ~MatchingProblem() = default;

  virtual size_t numApples() const = 0;
  virtual size_t numBananas() const = 0;

  /// Get a short list of candidates for all banana indices.
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
  /// \param[out] candidates_for_bananas Candidates from the Apples-list that could potentially
  ///                                    match for each banana.
  virtual inline void getCandidates(CandidatesList* candidates_for_bananas) {
    CHECK_NOTNULL(candidates_for_bananas)->clear();
    const size_t num_bananas = numBananas();
    candidates_for_bananas->resize(num_bananas);
    for (size_t banana_idx = 0u; banana_idx < num_bananas; ++banana_idx) {
      getAppleCandidatesForBanana(
          banana_idx, &(*candidates_for_bananas)[banana_idx]);
    }
  }

  /// Get a short list of candidates for a given banana index.
  ///
  /// \param[in] banana_index The index of the banana queried for candidates.
  /// \param[out] candidates_for_bananas Candidates from the Apples-list that could potentially
  ///                                    match for each element of Bananas.
  virtual void getAppleCandidatesForBanana(int /*banana_index*/, Candidates* /*candidates*/) {
    LOG(FATAL) << "Not implemented.";
  }

  /// Gets called at the beginning of the matching problem; i.e. to setup kd-trees, lookup tables
  /// or the like.
  virtual bool doSetup() = 0;

  /// List of tested match pairs for every banana. This is only retrieved and stored if the
  /// flag 'matcher_store_all_tested_pairs' is set to true.
  CandidatesList all_tested_pairs_;
};
}  // namespace aslam
#endif //ASLAM_CV_MATCHING_PROBLEM_H_
