#ifndef ASLAM_CAMERAS_DISTORTION_MAP_H_
#define ASLAM_CAMERAS_DISTORTION_MAP_H_

#include <Eigen/Core>
#include <glog/logging.h>

namespace aslam {

/// \class Distortion map
/// \brief For most distortion models (e.g. equidistant distortion), the
///     undistortion is not available in closed form. The undistortion map can
///     be used to once calculate the corresponding undistorted pixel location
///     and then look it up later.
class DistortionMap {
 public:
  DistortionMap()
      : fully_calculated_(false){};

  DistortionMap(const unsigned& image_width, const unsigned& image_height)
      : image_width_(image_width),
        image_height_(image_height),
        distortion_map_(image_width * image_height),
        fully_calculated_(false){};

  void set(
      const unsigned& col, const unsigned& row, const Eigen::Vector2d& point) {
    distortion_map_[row * image_width_ + col] = point;
  }
  Eigen::Vector2d& get(
      const unsigned& col, const unsigned& row, Eigen::Vector2d* point) const {
    CHECK_LE(row, image_height_);
    CHECK_LE(col, image_width_);
    CHECK(fully_calculated_) << "Calculate distortion map before using it.";
    *point = distortion_map_[row * image_width_ + col];
  }
  bool available() const {
    return fully_calculated_;
  }
  void calculated() {
    fully_calculated_ = true;
  }

 private:
  std::vector<Eigen::Vector2d> distortion_map_;
  unsigned image_width_;
  unsigned image_height_;
  bool fully_calculated_;
};

}  // namespace aslam

#endif  // ASLAM_CAMERAS_DISTORTION_MAP_H_
