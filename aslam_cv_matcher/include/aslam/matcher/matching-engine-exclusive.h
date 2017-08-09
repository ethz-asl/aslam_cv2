#ifndef ASLAM_CV_MATCHINGENGINE_EXCLUSIVE_H_
#define ASLAM_CV_MATCHINGENGINE_EXCLUSIVE_H_
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include <aslam/common/macros.h>
#include <aslam/common/memory.h>
#include <aslam/common/timer.h>
#include <aslam/matcher/match.h>

#include "aslam/matcher/matching-engine.h"

/// \addtogroup Matching
/// @{
///
/// @}

namespace aslam {

/// \brief Matches apples with bananas, such that the resulting matches are exclusive, i.e.
///        every banana matches to at most one apple and vice versa. Not every banana might find
///        a matching apple and vice versa. The assignment procedure is greedy and recursive.
///        Iterating over all bananas, starting at banana 0, the best apple is assigned,
///        replacing a previous assignment to this apple iff the current match score is higher.
///        The banana from the replaced matched is then recursively reassigned to the next best
///        apple (if there is any).
template<typename MatchingProblem>
class MatchingEngineExclusive : public MatchingEngine<MatchingProblem> {
 public:
  using MatchingEngine<MatchingProblem>::match;
  FRIEND_TEST(PriorityMatchingTest, TestAssignBest);

  ASLAM_POINTER_TYPEDEFS(MatchingEngineExclusive);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingEngineExclusive);

  MatchingEngineExclusive() {};
  virtual ~MatchingEngineExclusive() {};

  virtual bool match(MatchingProblem* problem,
                     typename MatchingProblem::MatchesWithScore* matches_A_B);

private:
  /// \brief Recursively assigns the next best apple to the given banana.
  inline void assignBest(int index_banana) {
    CHECK_GE(index_banana, 0);

    // Iterate through the next best apple candidates to find the next best fit (if any).
    for (; iterator_to_next_best_apple_[index_banana] != candidates_[index_banana].end();
        ++(iterator_to_next_best_apple_[index_banana])) {
      const size_t next_best_apple_for_this_banana =
          iterator_to_next_best_apple_[index_banana]->index_apple;
      CHECK_LT(next_best_apple_for_this_banana, temporary_matches_.size());

      // Write access to the next candidate.
      typename MatchingProblem::Candidate& temporary_candidate =
          temporary_matches_[next_best_apple_for_this_banana];

      if (temporary_candidate.index_apple < 0) {
        // Apple is still available. Assign the current candidate to this apple.
        temporary_candidate = *iterator_to_next_best_apple_[index_banana];
        break;
      } else if (temporary_candidate < *iterator_to_next_best_apple_[index_banana]) {
        // Apple is already assigned, but this one is better.
        const int lonely_banana = temporary_candidate.index_banana;
        temporary_candidate = *iterator_to_next_best_apple_[index_banana];
        // Recursively look for an alternative for the lonely banana.
        assignBest(lonely_banana);
        break;
      }
    }
  }

  /// \brief List of sorted candidates for a given banana. (i.e. candidates_[banana_index] refers
  ///        to a list of apples sorted wrt. the matching score (note: it's sorted bottom-up, so
  ///        .begin() points to the worst apple match for this banana. use .rbegin() instead to get
  ///        best candidate.)
  typename MatchingProblem::CandidatesList candidates_;

  /// \brief The temporary matches assigned to each apple. (i.e. temporary_matches_[apple_index]
  ///        returns the current match for this apple. May change during the assignment procedure.
  typename MatchingProblem::Candidates temporary_matches_;

  /// \brief Iterators to the next best apple candidate for each banana.
  ///        (i.e. iterator_to_next_best_apple_[banana_index] points to the next best candidate
  ///        for the given banana. Points to candiates_[banana_index].end() if no next best
  ///        candidate available.
  Aligned<std::vector, typename MatchingProblem::Candidates::iterator>
      iterator_to_next_best_apple_;
};

template<typename MatchingProblem>
bool MatchingEngineExclusive<MatchingProblem>::match(
    MatchingProblem* problem, typename MatchingProblem::MatchesWithScore* matches_A_B) {
  timing::Timer method_timer("MatchingEngineExclusive<MatchingProblem>::match()");

  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(matches_A_B);
  matches_A_B->clear();

  if (problem->doSetup()) {
    const size_t num_bananas = problem->numBananas();
    const size_t num_apples = problem->numApples();

    problem->getCandidates(&candidates_);
    CHECK_EQ(candidates_.size(), num_bananas) << "The size of the candidates list does not "
        << "match the number of bananas of the problem. getCandidates(...) of the given matching "
        << "problem is supposed to return a vector of candidates for each banana and hence the "
        << "size of the returned vector must match the number of bananas.";

    temporary_matches_.clear();
    temporary_matches_.resize(num_apples);

    iterator_to_next_best_apple_.resize(num_bananas);

    // Collect all apple candidates for every banana.
    for (size_t index_banana = 0; index_banana < num_bananas; ++index_banana) {
      // Sorts the candidates in descending order.
      std::sort(candidates_[index_banana].begin(), candidates_[index_banana].end(),
                std::greater<typename MatchingProblem::Candidate>());

      iterator_to_next_best_apple_[index_banana] = candidates_[index_banana].begin();
    }

    // Find the best apple for every banana.
    for (size_t index_banana = 0; index_banana < num_bananas; ++index_banana) {
      assignBest(index_banana);
    }

    // Assign the exclusive matches to the match vector.
    for (const typename MatchingProblem::Candidate& candidate : temporary_matches_)  {
      if (candidate.index_apple >= 0) {
        CHECK_LT(candidate.index_apple, static_cast<int>(num_apples));
        CHECK_GE(candidate.index_banana, 0);
        CHECK_LT(candidate.index_banana, static_cast<int>(num_bananas));

        matches_A_B->emplace_back(candidate.index_apple, candidate.index_banana, candidate.score);
      }
    }

    method_timer.Stop();
    return true;
  } else {
    LOG(ERROR) << "Setting up the matching problem (.doSetup()) failed.";
    method_timer.Stop();
    return false;
  }
}

}  // namespace aslam
#endif //ASLAM_CV_MATCHINGENGINE_EXCLUSIVE_H_
