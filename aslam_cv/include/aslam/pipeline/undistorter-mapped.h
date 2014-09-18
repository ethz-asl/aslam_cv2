#ifndef ASLAM_MAPPED_UNDISTORTER_H_
#define ASLAM_MAPPED_UNDISTORTER_H_

#include "undistorter.h"

namespace aslam {

/// \class MappedUndistorter
/// \brief A class that encapsulates image undistortion for building frames from images.
///
/// This class utilizes the OpenCV remap() function:
/// http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html?highlight=remap#remap
class MappedUndistorter : public Undistorter {
public:
  ASLAM_POINTER_TYPEDEFS(MappedUndistorter);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MappedUndistorter);

  MappedUndistorter();

  /// \brief Create an undistorter.
  ///
  /// The interpolation types are from OpenCV:
  ///
  /// - cv::INTER_NEAREST  - A nearest-neighbor interpolation.
  /// - cv::INTER_LINEAR   - A bilinear interpolation (used by default).
  /// - cv::INTER_AREA     - Resampling using pixel area relation. It may be a
  ///                        preferred method for image decimation, as it gives
  ///                        moire-free results.  But when the image is zoomed,
  ///                        it is similar to the INTER_NEAREST method.
  /// - cv::INTER_CUBIC    - A bicubic interpolation over 4x4 pixel neighborhood.
  /// - cv::INTER_LANCZOS4 - A Lanczos interpolation over 8x8 pixel neighborhood.
  ///
  /// Map matrices (map_x and map_y) must be the size of the output camera geometry.
  /// This will be checked by the constructor.
  ///
  /// \param[in] input_camera  The camera intrinsics for the original image.
  /// \param[in] output_camera The camera intrinsics after undistortion.
  /// \param[in] map_x         The map from input to output x coordinates.
  /// \param[in] map_y         The map from input to output y coordinates.
  /// \param[in] interpolation_method An enum specifying the interpolation types.
  /// \param[in] copy_image    Should we copy the input image?
  MappedUndistorter(const std::shared_ptr<Camera>& input_camera,
                    const std::shared_ptr<Camera>& output_camera,
                    const cv::Mat& map_x, const cv::Mat& map_y, int interpolation);

  virtual ~MappedUndistorter();

  /// \brief Produce an undistorted image from an input image.
  virtual void undistortImage(const cv::Mat& input_image, cv::Mat* output_image) const;

private:
  /// \brief LUT for x coordinates.
  cv::Mat map_x_;
  /// \brief LUT for y coordinates.
  cv::Mat map_y_;
  /// \brief Interpolation strategy
  int interpolation_method_;
};

}  // namespace aslam

#endif // ASLAM_UNDISTORTER_H_
