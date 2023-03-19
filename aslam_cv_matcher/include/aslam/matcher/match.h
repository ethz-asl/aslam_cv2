#ifndef ASLAM_MATCH_H_
#define ASLAM_MATCH_H_

#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace aslam {

/// \brief A struct to encapsulate a match between two lists.
///        The matches are indices into these lists.
struct FrameToFrameMatch : std::pair<size_t, size_t> {
  FrameToFrameMatch() = default;
  FrameToFrameMatch(size_t first_, size_t second_)
      : std::pair<size_t, size_t>(first_, second_) {}
  virtual ~FrameToFrameMatch() = default;
  size_t getKeypointIndexInFrameA() const {
    return first;
  }
  void setKeypointIndexInFrameA(size_t first_) {
    first = first_;
  }
  size_t getKeypointIndexInFrameB() const {
    return second;
  }
  void setKeypointIndexInFrameB(size_t second_) {
    second = second_;
  }
};

typedef Aligned<std::vector, FrameToFrameMatch> FrameToFrameMatches;
typedef Aligned<std::vector, FrameToFrameMatches> FrameToFrameMatchesList;

}  // namespace aslam

#endif  // ASLAM_MATCH_H_
