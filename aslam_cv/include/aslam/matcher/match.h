#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

#include <vector>

namespace aslam {

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
struct MatchWithScore {
  /// \brief Initialize to an invalid match.
  MatchWithScore()
      : correspondence {-1, -1}, score(0.0) {}

  /// \brief Initialize with correspondences and a score.
  MatchWithScore(int index_apple, int index_banana, double _score)
      : correspondence {index_apple, index_banana}, score(_score) {}

  /// \brief Get the index into list A.
  int getIndexApple() const {
    return correspondence[0];
  }

  /// \brief Get the index into list B.
  int getIndexBanana() const {
    return correspondence[1];
  }

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

  int correspondence[2];
  double score;
};

typedef std::vector<MatchWithScore> MatchesWithScore;
typedef std::pair<size_t, size_t> Match;
typedef std::vector<Match> Matches;

}  // namespace aslam

#endif // ASLAM_MATCH_H_
