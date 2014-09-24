#ifndef ASLAM_MAPPED_UNDISTORTER_H_
#define ASLAM_MAPPED_UNDISTORTER_H_

#include <aslam/pipeline/undistorter.h>
#include <aslam/cameras/camera.h>

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

protected:
  MappedUndistorter();

public:
  /// \brief Create a mapped undistorter using externally provided maps.
  ///
  /// The interpolation types are from OpenCV:
  /// - cv::INTER_NEAREST  - A nearest-neighbor interpolation.
  /// - cv::INTER_LINEAR   - A bilinear interpolation (used by default).
  /// - cv::INTER_AREA     - Resampling using pixel area relation. It may be a
  ///                        preferred method for image decimation, as it gives
  ///                        moire-free results.  But when the image is zoomed,
  ///                        it is similar to the INTER_NEAREST method.
  /// - cv::INTER_CUBIC    - A bicubic interpolation over 4x4 pixel neighborhood.
  /// - cv::INTER_LANCZOS4 - A Lanczos interpolation over 8x8 pixel neighborhood.
  ///
  /// Map matrices (map_u and map_v) must be the size of the output camera geometry.
  /// This will be checked by the constructor.
  ///
  /// \param[in] input_camera  The camera intrinsics for the original image.
  /// \param[in] output_camera The camera intrinsics after undistortion.
  /// \param[in] map_u         The map from input to output u coordinates.
  /// \param[in] map_v         The map from input to output v coordinates.
  /// \param[in] interpolation An enum specifying the interpolation types.
  MappedUndistorter(aslam::Camera::Ptr input_camera, aslam::Camera::Ptr output_camera,
                    const cv::Mat& map_u, const cv::Mat& map_v, int interpolation);

  virtual ~MappedUndistorter() = default;

  /// \brief Produce an undistorted image from an input image.
  virtual void processImage(const cv::Mat& input_image, cv::Mat* output_image) const;

  /// Get the undistorter map for the u-coordinate.
  const cv::Mat& getUndistortMapU() const { return map_u_; };

  /// Get the undistorter map for the u-coordinate.
  const cv::Mat& getUndistortMapV() const { return map_v_; };

private:
  /// \brief LUT for u coordinates.
  const cv::Mat map_u_;
  /// \brief LUT for v coordinates.
  const cv::Mat map_v_;
  /// \brief Interpolation strategy
  int interpolation_method_;
};

}  // namespace aslam

#endif // ASLAM_UNDISTORTER_H_
