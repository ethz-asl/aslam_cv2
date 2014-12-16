#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

#include <vector>

namespace aslam {

enum CandidateScore {
  Approximate,
  Final
};

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
struct Match {
  /// \brief Initialize to an invalid match.
  Match()
      : correspondence {-1, -1}, score(0.0) {}

  /// \brief Initialize with correspondences and a score.
  Match(int index_apple, int index_banana, double _score)
      : correspondence {index_apple, index_banana}, score(_score) {}

  /// \brief Get the index into list A.
  int getIndexApple() const {
    return correspondence[0];
  }

  /// \brief Get the index into list B.
  int getIndexBanana() const {
    return correspondence[1];
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
    return (this->correspondence[0] == other.correspondence[0]) &&
           (this->correspondence[1] == other.correspondence[1]) &&
           (this->score == other.score);
  }

  int correspondence[2];
  double score;
};

typedef std::vector<Match> Matches;

}  // namespace aslam

#endif // ASLAM_MATCH_H_
