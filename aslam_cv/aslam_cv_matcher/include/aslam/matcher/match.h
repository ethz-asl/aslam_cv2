#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

#include <vector>

#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/stl-helpers.h>
#include <Eigen/Core>
#include <glog/logging.h>
#include <opencv2/features2d/features2d.hpp>

namespace aslam {
class VisualFrame;
class VisualNFrame;

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
struct MatchWithScore;
typedef Aligned<std::vector, MatchWithScore>::type MatchesWithScore;
typedef std::pair<size_t, size_t> Match;
typedef Aligned<std::vector, Match>::type Matches;
typedef Aligned<std::vector, Matches>::type MatchesList;
typedef Aligned<std::vector, cv::DMatch>::type OpenCvMatches;

struct MatchWithScore {
  template<typename MatchWithScore, typename Match>
  friend void convertMatches(
      const typename Aligned<std::vector, MatchWithScore>::type& matches_with_score_A_B,
      typename Aligned<std::vector, Match>::type* matches_A_B);
  template<typename MatchWithScore>
  friend void convertMatches(
      const typename Aligned<std::vector, MatchWithScore>::type& matches_with_score_A_B,
      OpenCvMatches* matches_A_B);

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
#define ASLAM_ADD_MATCH_TYPEDEFS_WITH_ALIASES(                                                    \
    MatchingProblemClass, getAppleIndexAlias, getBananaIndexAlias)                                \
  struct MatchingProblemClass ## MatchWithScore : public aslam::MatchWithScore {                  \
    MatchingProblemClass ## MatchWithScore(int index_apple, int index_banana, double _score)      \
        : aslam::MatchWithScore(index_apple, index_banana, _score) {}                             \
    virtual ~MatchingProblemClass ## MatchWithScore() = default;                                  \
    int getAppleIndexAlias() const {                                                              \
      return aslam::MatchWithScore::getIndexApple();                                              \
    }                                                                                             \
    int getBananaIndexAlias() const {                                                             \
      return aslam::MatchWithScore::getIndexBanana();                                             \
    }                                                                                             \
  };                                                                                              \
  typedef MatchingProblemClass ## MatchWithScore MatchWithScore;                                  \
  typedef Aligned<std::vector, MatchingProblemClass ## MatchWithScore>::type MatchesWithScore;    \
  typedef Aligned<std::vector, MatchesWithScore>::type MatchesWithScoreList;                      \
  struct MatchingProblemClass ## Match : public aslam::Match {                                    \
    MatchingProblemClass ## Match() = default;                                                    \
    MatchingProblemClass ## Match(size_t first_, size_t second_) : aslam::Match(first_, second_){}\
    virtual ~MatchingProblemClass ## Match() = default;                                           \
    size_t getAppleIndexAlias() const { return first; }                                           \
    size_t getBananaIndexAlias() const { return second; }                                         \
  };                                                                                              \
  typedef MatchingProblemClass ## Match Match;                                                    \
  typedef Aligned<std::vector, MatchingProblemClass ## Match>::type Matches;                      \
  typedef Aligned<std::vector, Matches>::type MatchesList;                                        \

#endif // ASLAM_MATCH_H_
