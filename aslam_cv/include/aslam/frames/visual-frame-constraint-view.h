#ifndef ASLAM_VISUAL_FRAMES_VIEW_H_
#define ASLAM_VISUAL_FRAMES_VIEW_H_

#include <memory>

#include <Eigen/Dense>

#include <aslam/common/macros.h>
#include <aslam/frames/visual-frame.h>

namespace aslam {

class VisualFrameConstraintViewIterator;


/// \class VisualFrameConstraintView
/// \brief This class implements a view that fits over a VisualFrame to conveniently access
///        processed frame data such as keypoints, descriptors and so on.
class VisualFrameConstraintView  {

 private:
  const VisualFrame& frame_;
  size_t current_idx_;

 public:
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(VisualFrameConstraintView);

  typedef VisualFrameConstraintView iterator;

  VisualFrameConstraintView(const VisualFrame& frame) :
    frame_(frame),
    current_idx_(0) {};

  VisualFrameConstraintView(const VisualFrame& frame, size_t idx) :
    frame_(frame),
    current_idx_(idx) {};

  ~VisualFrameConstraintView() {};

 public:
  iterator operator[](std::size_t idx) {
    return iterator(frame, idx);
  }

  iterator& operator++() {
      ++current_idx_;
      return *this;
  }

  iterator operator++(int) {
      iterator orig = *this;
      ++(*this);
      return orig;
  }

  Eigen::Vector2d keypoint() {
    CHECK(frame_.hasKeypointMeasurements());
    CHECK_LT(current_idx_, frame_.getKeypointMeasurement().rows());
    return frame_.getKeypointMeasurementUncertainties().row(current_idx_);
  }


  Eigen::Vector2d& keypointUncertainty() {
    CHECK(frame_.hasKeypointMeasurementUncertainties());
    CHECK_LT(current_idx_, frame_.getKeypointMeasurementUncertainties().rows());
    return frame_.getKeypointMeasurementUncertainties().row(current_idx_);
  }


  unsigned char* descriptor() {
    CHECK(frame_.hasKeypointMeasurements());
    return
  }

  unsigned char* descriptorSizeBytes() {


  }


  //operator++
  //operator--
  //operator


};



//std::vector<int> a;
//std::vector<int>::iterator it(a);


}  // namespace aslam
#endif  // ASLAM_VISUAL_FRAMES_VIEW_H_
