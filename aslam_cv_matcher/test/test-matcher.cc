#include <algorithm>
#include <cmath>
#include <vector>

#include <aslam/common/entrypoint.h>
#include <aslam/matcher/match.h>
#include <aslam/matcher/matching-engine-exclusive.h>
#include <aslam/matcher/matching-engine-greedy.h>
#include <aslam/matcher/matching-problem.h>
#include <gtest/gtest.h>

namespace aslam {

class SimpleMatchProblem : public aslam::MatchingProblem {

  std::vector<double> apples_;
  std::vector<double> bananas_;

  aslam::Matches matches_A_B_;

 public:
  SimpleMatchProblem() {}
  ~SimpleMatchProblem() {}
  typedef aslam::MatchWithScore MatchWithScore;
  typedef Aligned<std::vector, MatchWithScore> MatchesWithScore;
  typedef aslam::Match Match;
  typedef Aligned<std::vector, Match> Matches;

  virtual size_t numApples() const {
    return apples_.size();
  }
  virtual size_t numBananas() const {
    return bananas_.size();
  }

  virtual bool doSetup() {
    return true;
  }

  template<typename iter>
  void setApples(const iter& first, const iter& last) {
    apples_.clear();
    apples_.insert(apples_.end(), first, last);
  }
  template<typename iter>
  void setBananas(const iter& first, const iter& last) {
    bananas_.clear();
    bananas_.insert(bananas_.end(), first, last);
  }

  void sortMatches() {
    std::sort(matches_A_B_.begin(),matches_A_B_.end());
  }

  virtual void getAppleCandidatesForBanana(int b, Candidates* candidates) {
     CHECK_NOTNULL(candidates);
     candidates->clear();

     // just returns all apples with no score
     for (unsigned int index_apple = 0; index_apple < numApples(); ++index_apple) {
       double score = -fabs(apples_[index_apple] - bananas_[b]);
       candidates->emplace_back(index_apple, b, score, 0);
     }
   };
};

TEST(PriorityMatchingTest, TestAssignBest) {
  ////////////////////
  ////// SCENARIO
  ////////////////////
  // bananas       apples, priority
  //    0               0, 0
  //    1               1, 1
  //    2               2, 0
  //    3               3, 1
  //
  // candidates:
  //    (b, a), score
  //    ------
  //    (0, 0), 0.0
  //    (1, 0), 1.0
  //    (1, 1), 2.0
  //    (1, 2), 3.0
  //    (2, 1), 4.0
  //    (2, 2), 5.0
  //    (3, 1), 6.0
  //    (3, 2), 7.0
  //    (3, 3), 0.5

  // The following behaviour is expected:
  // 1. banana 0 is assigned to apple 0 (only candidate)
  // 2. banana 1 is assigned to apple 1 (priority of apple 1 outrules score of apple 2)
  // 3. banana 2 is assigned to apple 1 (again, priority outrules score of apple 2)
  // 4. banana 1 is reassigned to apple 2 (next best after apple 1 for banana 1)
  // 5. banana 3 is assigned to apple 1 (score outrules apple 3 with same priority as apple 1)
  // 6. banana 2 is reassigned to apple 2 (next best)
  // 7. banana 1 is reassigned to apple 0 (next best)
  // 8. banana 0 loses its assignment apple 0 (not reassigned again since no other candidates)

  // This leads to the following solution:
  // (a, b)
  // ------
  // (0, 1)
  // (1, 3)
  // (2, 2)
  aslam::MatchingEngineExclusive<SimpleMatchProblem> matching_engine;

  matching_engine.candidates_.resize(4);
  matching_engine.iterator_to_next_best_apple_.resize(4);
  matching_engine.temporary_matches_.resize(4);

  matching_engine.candidates_[0].emplace_back(0, 0, 0.0, 0);
  matching_engine.iterator_to_next_best_apple_[0] = matching_engine.candidates_[0].begin();

  matching_engine.candidates_[1].emplace_back(0, 1, 1.0, 0);
  matching_engine.candidates_[1].emplace_back(1, 1, 2.0, 1);
  matching_engine.candidates_[1].emplace_back(2, 1, 3.0, 0);
  std::sort(matching_engine.candidates_[1].begin(), matching_engine.candidates_[1].end(),
            std::greater<aslam::MatchingProblem::Candidate>());
  matching_engine.iterator_to_next_best_apple_[1] = matching_engine.candidates_[1].begin();

  matching_engine.candidates_[2].emplace_back(1, 2, 4.0, 1);
  matching_engine.candidates_[2].emplace_back(2, 2, 5.0, 0);
  std::sort(matching_engine.candidates_[2].begin(), matching_engine.candidates_[2].end(),
            std::greater<aslam::MatchingProblem::Candidate>());
  matching_engine.iterator_to_next_best_apple_[2] = matching_engine.candidates_[2].begin();

  matching_engine.candidates_[3].emplace_back(1, 3, 6.0, 1);
  matching_engine.candidates_[3].emplace_back(2, 3, 7.0, 0);
  matching_engine.candidates_[3].emplace_back(3, 3, 0.5, 1);
  std::sort(matching_engine.candidates_[3].begin(), matching_engine.candidates_[3].end(),
            std::greater<aslam::MatchingProblem::Candidate>());
  matching_engine.iterator_to_next_best_apple_[3] = matching_engine.candidates_[3].begin();

  for (size_t i = 0; i < 4; ++i) {
    matching_engine.assignBest(i);
  }

  EXPECT_EQ(matching_engine.temporary_matches_[0].index_apple, 0);
  EXPECT_EQ(matching_engine.temporary_matches_[0].index_banana, 1);

  EXPECT_EQ(matching_engine.temporary_matches_[1].index_apple, 1);
  EXPECT_EQ(matching_engine.temporary_matches_[1].index_banana, 3);

  EXPECT_EQ(matching_engine.temporary_matches_[2].index_apple, 2);
  EXPECT_EQ(matching_engine.temporary_matches_[2].index_banana, 2);

  EXPECT_EQ(matching_engine.temporary_matches_[3].index_apple, -1);
  EXPECT_EQ(matching_engine.temporary_matches_[3].index_banana, -1);
}

TEST(TestMatcherExclusive, EmptyMatch) {
  SimpleMatchProblem mp;
  aslam::MatchingEngineExclusive<SimpleMatchProblem> me;

  SimpleMatchProblem::MatchesWithScore matches;
  me.match(&mp, &matches);
  EXPECT_TRUE(matches.empty());

  matches.clear();
  std::vector<float> bananas { 1.1, 2.2, 3.3 };
  mp.setBananas(bananas.begin(), bananas.end());
  me.match(&mp, &matches);
  EXPECT_TRUE(matches.empty());
}

TEST(TestMatcherExclusive, ExclusiveMatcher) {
  std::vector<float> apples( { 1.1, 2.2, 3.3, 4.4, 5.5 });
  std::vector<float> bananas = { 1.0, 2.0, 3.0, 4.0, 5.0, 1.1 };
  std::vector<int> banana_index_for_apple = { 5, 1, 2, 3, 4 };

  SimpleMatchProblem match_problem;
  aslam::MatchingEngineExclusive<SimpleMatchProblem> matching_engine;

  match_problem.setApples(apples.begin(), apples.end());
  EXPECT_EQ(5u, match_problem.numApples());

  SimpleMatchProblem::MatchesWithScore matches;
  matching_engine.match(&match_problem, &matches);
  EXPECT_TRUE(matches.empty());

  match_problem.setBananas(bananas.begin(), bananas.end());
  EXPECT_EQ(6u, match_problem.numBananas());

  matches.clear();
  matching_engine.match(&match_problem, &matches);
  EXPECT_EQ(5u, matches.size());

  match_problem.sortMatches();

  for (const SimpleMatchProblem::MatchWithScore &match : matches) {
    EXPECT_EQ(match.getIndexBanana(), banana_index_for_apple[match.getIndexApple()]);
  }
}

TEST(TestMatcher, EmptyMatch) {
  SimpleMatchProblem mp;
  aslam::MatchingEngineGreedy<SimpleMatchProblem> me;

  SimpleMatchProblem::MatchesWithScore matches;
  me.match(&mp, &matches);
  EXPECT_TRUE(matches.empty());

  matches.clear();
  std::vector<float> bananas { 1.1, 2.2, 3.3 };
  mp.setBananas(bananas.begin(), bananas.end());
  me.match(&mp, &matches);
  EXPECT_TRUE(matches.empty());
}

TEST(TestMatcher, GreedyMatcher) {

  std::vector<float> apples( { 1.1, 2.2, 3.3, 4.4, 5.5 });
  std::vector<float> bananas = { 1.0, 2.0, 3.0, 4.0, 5.0, 0.0 };
  std::vector<int> ind_a_of_b = { 0, 1, 2, 3, 4, -1 };

  SimpleMatchProblem mp;
  aslam::MatchingEngineGreedy<SimpleMatchProblem> me;

  mp.setApples(apples.begin(), apples.end());
  EXPECT_EQ(5u, mp.numApples());

  SimpleMatchProblem::MatchesWithScore matches;
  me.match(&mp, &matches);
  EXPECT_TRUE(matches.empty());

  mp.setBananas(bananas.begin(), bananas.end());
  EXPECT_EQ(6u, mp.numBananas());

  matches.clear();
  me.match(&mp, &matches);
  EXPECT_EQ(5u, matches.size());

  mp.sortMatches();

  for (const SimpleMatchProblem::MatchWithScore& match : matches) {
    EXPECT_EQ(match.getIndexApple(), ind_a_of_b[match.getIndexBanana()]);
  }
}

}  // namespace aslam

ASLAM_UNITTEST_ENTRYPOINT
