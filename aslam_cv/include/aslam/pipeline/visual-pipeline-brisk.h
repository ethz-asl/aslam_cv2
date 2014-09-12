#ifndef ASLAM_BRISK_PIPELINE_H_
#define ASLAM_BRISK_PIPELINE_H_

#include <aslam/pipeline/visual-pipeline.h>
#include <aslam/pipeline/visual-pipeline-null.h>

namespace cv {
class FeatureDetector;
class DescriptorExtractor;
}  // namespace cv

namespace aslam {

/// \class BriskVisualPipeline
/// \brief A visual pipeline to extract Brisk features.
class BriskVisualPipeline : public VisualPipeline {
public:
  ASLAM_POINTER_TYPEDEFS(BriskVisualPipeline);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(BriskVisualPipeline);

  BriskVisualPipeline();

  /// \brief Initialize the brisk pipeline with a camera.
  ///
  /// \param[in] camera             The intrinsic calibration of this camera.
  /// \param[in] copy_images        Should we deep copy the images passed in?
  /// \param[in] octaves            Number of octaves Brisk should process.
  /// \param[in] uniformity_radius  Uniformity radius Brisk should use.
  /// \param[in] absolute_threshold The brisk absolute threshold.
  ///                               Low makes more keypoints.
  /// \param[in] max_number_of_keypoints The maximum number of keypoints to return.
  /// \param[in] rotation_invariant Should Brisk account for keypoint rotation?
  /// \param[in] scale_invariant    Should Brisk account for keypoint scale?
  BriskVisualPipeline(const std::shared_ptr<Camera>& camera, bool copy_images,
                      size_t octaves, double uniformity_radius,
                      double absolute_threshold, size_t max_number_of_keypoints,
                      bool rotation_invariant, bool scale_invariant);

  /// \brief Initialize the brisk pipeline with a preprocessing pipeline.
  ///
  /// \param[in] preprocessing      A visual pipeline to do preprocessing such as
  ///                               contrast enhancement or undistortion.
  /// \param[in] octaves            Number of octaves Brisk should process.
  /// \param[in] uniformity_radius  Uniformity radius Brisk should use.
  /// \param[in] absolute_threshold The brisk absolute threshold.
  ///                               Low makes more keypoints.
  /// \param[in] max_number_of_keypoints The maximum number of keypoints to return.
  /// \param[in] rotation_invariant Should Brisk account for keypoint rotation?
  /// \param[in] scale_invariant    Should Brisk account for keypoint scale?
  BriskVisualPipeline(const VisualPipeline::Ptr& preprocessing,
                      size_t octaves, double uniformity_radius,
                      double absolute_threshold, size_t max_number_of_keypoints,
                      bool rotation_invariant, bool scale_invariant);

  virtual ~BriskVisualPipeline();

  /// \brief Initialize the brisk pipeline.
  ///
  /// \param[in] octaves            Number of octaves Brisk should process.
  /// \param[in] uniformity_radius  Uniformity radius Brisk should use.
  /// \param[in] absolute_threshold The brisk absolute threshold.
  ///                               Low makes more keypoints.
  /// \param[in] max_number_of_keypoints The maximum number of keypoints to return.
  /// \param[in] rotation_invariant Should Brisk account for keypoint rotation?
  /// \param[in] scale_invariant    Should Brisk account for keypoint scale?
  void initializeBrisk(size_t octaves, double uniformity_radius,
                       double absolute_threshold, size_t max_number_of_keypoints,
                       bool rotation_invariant, bool scale_invariant);

protected:
  /// \brief Process the frame and fill the results into the frame variable
  ///
  /// The top level function will already fill in the timestamps and the output camera.
  /// \param[in]     image The image data.
  /// \param[in/out] frame The visual frame. This will be constructed before calling.
  virtual void processFrame(const cv::Mat& image,
                            std::shared_ptr<VisualFrame>* frame) const;
private:
  VisualPipeline::Ptr preprocessing_;
  std::shared_ptr<cv::FeatureDetector> detector_;
  std::shared_ptr<cv::DescriptorExtractor> extractor_;

  size_t octaves_;
  double uniformity_radius_;
  double absolute_threshold_;
  size_t max_number_of_keypoints_;
  bool rotation_invariant_;
  bool scale_invariant_;
};

}  // namespace aslam

#endif // ASLAM_BRISK_PIPELINE_H_
