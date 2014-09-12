#ifndef ASLAM_UNDISTORTER_H_
#define ASLAM_UNDISTORTER_H_

#include <opencv2/core/core.hpp>

#include "visual-pipeline.h"

namespace aslam {

/// \class Undistorter
/// \brief A class that encapsulates image undistortion for building frames from images.
///
/// This class utilizes the OpenCV remap() function:
/// http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html?highlight=remap#remap
class Undistorter : public VisualPipeline {
public:
  ASLAM_POINTER_TYPEDEFS(Undistorter);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Undistorter);

  Undistorter();

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
  /// \param[in] input_camera  The camera intrinsics for the original image.
  /// \param[in] output_camera The camera intrinsics after undistortion.
  /// \param[in] map_x         The map from input to output x coordinates.
  /// \param[in] map_y         The map from input to output y coordinates.
  /// \param[in] interpolation An enum specifying the interpolation types.
  Undistorter(const std::shared_ptr<Camera>& input_camera,
              const std::shared_ptr<Camera>& output_camera,
              const cv::Mat& map_x, const cv::Mat& map_y, int interpolation);

  virtual ~Undistorter();

  /// \brief Produce an undistorted image from an input image.
  void undistortImage(const cv::Mat& input_image, cv::Mat* output_image) const;

protected:
  /// \brief Process the frame and fill the results into the frame variable
  ///
  /// The top level function will already fill in the timestamps and the output camera.
  /// \param[in]     image The image data.
  /// \param[in/out] frame The visual frame. This will be constructed before calling.
  virtual void processFrame(const cv::Mat& image,
                            std::shared_ptr<VisualFrame>* frame) const;
private:
  /// \brief LUT for x coordinates.
  cv::Mat map_x_;
  /// \brief LUT for y coordinates.
  cv::Mat map_y_;
  /// \brief Interpolation strategy
  int interpolation_;
};

}  // namespace aslam

#endif // ASLAM_UNDISTORTER_H_
