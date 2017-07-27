#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

#include <vector>

#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/stl-helpers.h>
#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/features2d/features2d.hpp>

namespace aslam {
class VisualFrame;
class VisualNFrame;

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
struct MatchWithScore;
typedef Aligned<std::vector, MatchWithScore> MatchesWithScore;
typedef std::pair<size_t, size_t> Match;
typedef Aligned<std::vector, Match> Matches;
typedef Aligned<std::vector, Matches> MatchesList;
typedef Aligned<std::vector, cv::DMatch> OpenCvMatches;

struct MatchWithScore {
  template <typename MatchWithScore, typename Match>
  friend void convertMatchesWithScoreToMatches(
      const Aligned<std::vector, MatchWithScore>& matches_with_score_A_B,
      Aligned<std::vector, Match>* matches_A_B);
  template <typename MatchWithScore>
  friend void convertMatchesWithScoreToOpenCvMatches(
      const Aligned<std::vector, MatchWithScore>& matches_with_score_A_B,
      OpenCvMatches* matches_A_B);
  FRIEND_TEST(TestMatcherExclusive, ExclusiveMatcher);
  FRIEND_TEST(TestMatcher, GreedyMatcher);
  template<typename MatchingProblem> friend class MatchingEngineGreedy;

  /// \brief Initialize to an invalid match.
  MatchWithScore() : correspondence {-1, -1}, score(0.0) {}

  /// \brief Initialize with correspondences and a score.
  MatchWithScore(int index_apple, int index_banana, double _score)
      : correspondence {index_apple, index_banana}, score(_score) {}

  void setIndexApple(int index_apple) {
    correspondence[0] = index_apple;
  }

  void setIndexBanana(int index_banana) {
      correspondence[1] = index_banana;
  }

  void setScore(double _score) {
    score = _score;
  }

  /// \brief Get the score given to the match.
  double getScore() const {
    return score;
  }

  bool operator<(const MatchWithScore &other) const {
    return this->score < other.score;
  }
  bool operator>(const MatchWithScore &other) const {
    return this->score > other.score;
  }

  bool operator==(const MatchWithScore& other) const {
    return (this->correspondence[0] == other.correspondence[0]) &&
           (this->correspondence[1] == other.correspondence[1]) &&
           (this->score == other.score);
  }

 protected:
  /// \brief Get the index into list A.
  int getIndexApple() const {
    return correspondence[0];
  }

  /// \brief Get the index into list B.
  int getIndexBanana() const {
    return correspondence[1];
  }

 private:
  int correspondence[2];
  double score;
};
}  // namespace aslam

// Macro that generates the match types for a given matching problem.
// getAppleIndexAlias and getBananaIndexAlias specify function aliases for retrieving the apple
// and banana index respectively of a match. These aliases should depend on how the apples and
// bananas are associated within the context of the given matching problem.

#define ASLAM_CREATE_MATCH_TYPES_WITH_ALIASES(                                                    \
    MatchType, getAppleIndexAlias, getBananaIndexAlias)                                           \
  struct MatchType ## MatchWithScore : public aslam::MatchWithScore {                             \
  MatchType ## MatchWithScore(int index_apple, int index_banana, double score)                    \
        : aslam::MatchWithScore(index_apple, index_banana, score) {}                              \
    virtual ~MatchType ## MatchWithScore() = default;                                             \
    int getAppleIndexAlias() const {                                                              \
      return aslam::MatchWithScore::getIndexApple();                                              \
    }                                                                                             \
    int getBananaIndexAlias() const {                                                             \
      return aslam::MatchWithScore::getIndexBanana();                                             \
    }                                                                                             \
  };                                                                                              \
  typedef Aligned<std::vector, MatchType ## MatchWithScore> MatchType ## MatchesWithScore;  \
  typedef Aligned<std::vector, MatchType ## MatchesWithScore>                               \
    MatchType ## MatchesWithScoreList;                                                            \
  struct MatchType ## Match : public aslam::Match {                                               \
      MatchType ## Match() = default;                                                             \
      MatchType ## Match(size_t first_, size_t second_) : aslam::Match(first_, second_){}         \
    virtual ~MatchType ## Match() = default;                                                      \
    size_t getAppleIndexAlias() const { return first; }                                           \
    size_t getBananaIndexAlias() const { return second; }                                         \
  };                                                                                              \
  typedef Aligned<std::vector, MatchType ## Match> MatchType ## Matches;                    \
  typedef Aligned<std::vector, MatchType ## Matches> MatchType ## MatchesList;              \

#define ASLAM_ADD_MATCH_TYPEDEFS(MatchType)                                                       \
  typedef MatchType ## MatchWithScore MatchWithScore;                                             \
  typedef Aligned<std::vector, MatchType ## MatchWithScore> MatchesWithScore;               \
  typedef Aligned<std::vector, MatchType ## MatchesWithScore> MatchesWithScoreList;         \
  typedef MatchType ## Match Match;                                                               \
  typedef Aligned<std::vector, MatchType ## Match> Matches;                                 \
  typedef Aligned<std::vector, MatchType ## Matches> MatchesList;                           \

namespace aslam {
ASLAM_CREATE_MATCH_TYPES_WITH_ALIASES(
    FrameToFrame, getKeypointIndexAppleFrame, getKeypointIndexBananaFrame);
ASLAM_CREATE_MATCH_TYPES_WITH_ALIASES(
    LandmarksToFrame, getKeypointIndex, getLandmarkIndex);
}  // namespace aslam

#endif // ASLAM_MATCH_H_
