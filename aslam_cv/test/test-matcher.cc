#include <aslam/common/entrypoint.h>

#include <math.h>
#include <vector>

#include <aslam/matcher/matching-engine-greedy.h>
#include <aslam/matcher/matching-problem.h>


class SimpleMatchProblem : public aslam::MatchingProblem<float> {

  std::vector<float> _apples;
  std::vector<float> _bananas;

  MatchesT _matches;

public:
  SimpleMatchProblem() {}
  ~SimpleMatchProblem() {}


  virtual int getLengthApples() const { return _apples.size(); }
  virtual int getLengthBananas() const { return _bananas.size(); }
  
  virtual float computeScore(int a, int b) {
    CHECK_LT(size_t(a),_apples.size());
    CHECK_LT(size_t(b),_bananas.size());
    return -fabs(_apples[a]-_bananas[b]);
  }

  virtual bool doSetup() {return true;}

  virtual void setBestMatches(const MatchesT &bestMatches) {
    _matches = bestMatches;
  }

  template <typename iter>
  void setApples(iter first, iter last) {
    _apples.clear();
    _apples.insert(_apples.end(),first,last);
  }
  template <typename iter>
  void setBananas(iter first, iter last) {
    _bananas.clear();
    _bananas.insert(_bananas.end(),first,last);
  }
  MatchesT &getMatches() { return _matches;}
};


class TestMatch : public testing::Test {
public:
  TestMatch() {}
  virtual ~TestMatch() {}

};

TEST(TestMatcher, EmptyMatch) {
  SimpleMatchProblem mp;
  aslam::MatchingEngineGreedy<SimpleMatchProblem> me;

  me.match(mp);
  EXPECT_EQ(0,mp.getMatches().size());

  std::vector<float> bananas{1.1,2.2,3.3};
  mp.setBananas(bananas.begin(),bananas.end());
  me.match(mp);
  EXPECT_EQ(0,mp.getMatches().size());
}

TEST(TestMatcher, GreedyMatcher) {

  std::vector<float> apples( {1.1, 2.2, 3.3, 4.4, 5.5});
  std::vector<float> bananas = {1.0, 2.0, 3.0, 4.0, 5.0, 0.0};
  std::vector<int> ind_a_of_b = {0, 1, 2, 3, 4, -1};

  SimpleMatchProblem mp;
  aslam::MatchingEngineGreedy<SimpleMatchProblem> me;


  mp.setApples(apples.begin(),apples.end());
  EXPECT_EQ(5,mp.getLengthApples());

  me.match(mp);
  EXPECT_EQ(0,mp.getMatches().size());

  mp.setBananas(bananas.begin(),bananas.end());
  EXPECT_EQ(6,mp.getLengthBananas());

  me.match(mp);
  EXPECT_EQ(5,mp.getMatches().size());

  std::sort(mp.getMatches().begin(),mp.getMatches().end());
  
  for(auto &match: mp.getMatches()) {
    EXPECT_EQ(match.getIndexA(), ind_a_of_b[match.getIndexB()]);
  }

}
