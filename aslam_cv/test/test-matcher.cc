#include <aslam/common/entrypoint.h>

#include <math.h>
#include <vector>
#include <algorithm>

#include <aslam/matcher/matching-engine-greedy.h>
#include <aslam/matcher/matching-problem.h>

class SimpleMatchProblem : public aslam::MatchingProblem<float> {

  std::vector<float> apples_;
  std::vector<float> bananas_;

  MatchesT matches_;

 public:
  SimpleMatchProblem() {
  }
  ~SimpleMatchProblem() {
  }

  virtual int getLengthApples() const {
    return apples_.size();
  }
  virtual int getLengthBananas() const {
    return bananas_.size();
  }

  virtual float computeScore(int a, int b) {
    CHECK_LT(size_t(a), apples_.size());
    CHECK_LT(size_t(b), bananas_.size());
    return -fabs(apples_[a] - bananas_[b]);
  }

  virtual bool doSetup() {
    return true;
  }

  virtual void setBestMatches(const MatchesT &bestMatches) {
    matches_ = bestMatches;
  }

  template<typename iter>
  void setApples(iter first, iter last) {
    apples_.clear();
    apples_.insert(apples_.end(), first, last);
  }
  template<typename iter>
  void setBananas(iter first, iter last) {
    bananas_.clear();
    bananas_.insert(bananas_.end(), first, last);
  }
  MatchesT &getMatches() {
    return matches_;
  }
};

class TestMatch : public testing::Test {
 public:
  TestMatch() {
  }
  virtual ~TestMatch() {
  }

};

TEST(TestMatcher, EmptyMatch) {
  SimpleMatchProblem mp;
  aslam::MatchingEngineGreedy<SimpleMatchProblem> me;

  me.match(mp);
  EXPECT_EQ(0u, mp.getMatches().size());

  std::vector<float> bananas { 1.1, 2.2, 3.3 };
  mp.setBananas(bananas.begin(), bananas.end());
  me.match(mp);
  EXPECT_EQ(0u, mp.getMatches().size());
}

TEST(TestMatcher, GreedyMatcher) {

  std::vector<float> apples( { 1.1, 2.2, 3.3, 4.4, 5.5 });
  std::vector<float> bananas = { 1.0, 2.0, 3.0, 4.0, 5.0, 0.0 };
  std::vector<int> ind_a_of_b = { 0, 1, 2, 3, 4, -1 };

  SimpleMatchProblem mp;
  aslam::MatchingEngineGreedy<SimpleMatchProblem> me;

  mp.setApples(apples.begin(), apples.end());
  EXPECT_EQ(5u, mp.getLengthApples());

  me.match(mp);
  EXPECT_EQ(0u, mp.getMatches().size());

  mp.setBananas(bananas.begin(), bananas.end());
  EXPECT_EQ(6, mp.getLengthBananas());

  me.match(mp);
  EXPECT_EQ(5u, mp.getMatches().size());

  std::sort(mp.getMatches().begin(), mp.getMatches().end());

  for (auto &match : mp.getMatches()) {
    EXPECT_EQ(match.getIndexA(), ind_a_of_b[match.getIndexB()]);
  }

}
