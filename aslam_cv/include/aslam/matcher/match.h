#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

namespace aslam {

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
template<typename Score>
struct Match {
  /// \brief Initialize to an invalid match.
  Match() : correspondence{ -1, -1 }, score(static_cast<Score>(0)) {}

  /// \brief Initialize with correspondences and a score.
  Match(int correspondence_a, int correspondence_b, Score score) :
    correspondence{correspondence_a, correspondence_b}, score(score) {}

    /// \brief Get the index into list A.
  int getIndexA()  const { return correspondence[0]; }

  /// \brief Get the index into list B.
  int getIndexB()  const { return correspondence[1]; }

  /// \brief Get the score given to the match.
  Score getScore() const { return score; }

  bool operator<( const Match &other) const {
    return this->score < other.score;
  }
  bool operator>( const Match &other) const {
    return this->score > other.score;
  }

  int correspondence[2];
  Score score;
};

}  // namespace aslam

#endif // ASLAM_MATCH_H_
