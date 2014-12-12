#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

#include <vector>

namespace aslam {

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
struct Match {
  /// \brief Initialize to an invalid match.
  Match()
      : index_apple(-1), index_banana(-1), score(0.0) {}

  /// \brief Initialize with correspondences and a score.
  Match(int _index_apple, int _index_banana, double _score)
      : index_apple(_index_apple), index_banana(_index_banana), score(_score) {}

  /// \brief Get the index into list A.
  int getIndexApple() const {
    return index_apple;
  }

  /// \brief Get the index into list B.
  int getIndexBanana() const {
    return index_banana;
  }

  /// \brief Get the score given to the match.
  double getScore() const {
    return score;
  }

  bool operator<(const Match &other) const {
    return this->score < other.score;
  }
  bool operator>(const Match &other) const {
    return this->score > other.score;
  }

  bool operator==(const Match& other) const {
    return (this->index_apple == other.index_apple) &&
        (this->index_banana == other.index_banana) &&
        (this->score == other.score);
  }

  int index_apple;
  int index_banana;
  double score;
};

typedef std::vector<Match> Matches;

}  // namespace aslam

#endif // ASLAM_MATCH_H_
