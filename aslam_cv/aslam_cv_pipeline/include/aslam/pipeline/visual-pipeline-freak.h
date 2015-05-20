#ifndef ASLAM_FREAK_PIPELINE_H_
#define ASLAM_FREAK_PIPELINE_H_

#include <aslam/pipeline/visual-pipeline.h>
#include <aslam/pipeline/visual-pipeline-null.h>

namespace cv {
class FeatureDetector;
class DescriptorExtractor;
}  // namespace cv

namespace aslam {

class Undistorter;

/// \class FreakVisualPipeline
/// \brief A visual pipeline to extract Freak features.
class FreakVisualPipeline : public VisualPipeline {
public:
  ASLAM_POINTER_TYPEDEFS(FreakVisualPipeline);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(FreakVisualPipeline);

protected:
  /// Constructor for serialization.
  FreakVisualPipeline();

public:
  /// \brief Initialize the freak pipeline with a camera.
  ///
  /// \param[in] camera             The intrinsic calibration of this camera.
  /// \param[in] copy_images        Should we deep copy the images passed in?
  /// \param[in] octaves            Number of octaves for brisk/freak scale computation.
  /// \param[in] uniformity_radius  Gets multiplied on the keypoint scale and determines
  ///                               the distance to the next neighboring keypoint.
  /// \param[in] absolute_threshold The harris absolute threshold.
  ///                               Low makes more keypoints.
  /// \param[in] max_number_of_keypoints The maximum number of keypoints to return.
  /// \param[in] rotation_invariant Should freak estimate the keypoint orientation?
  /// \param[in] scale_invariant    Should freak estimate the keypoint scale?
  /// \param[in] pattern_scale Scale of the pattern for the freak feature descriptor.
  FreakVisualPipeline(const Camera::ConstPtr& camera, bool copy_images, size_t octaves,
                      double uniformity_radius, double absolute_threshold,
                      size_t max_number_of_keypoints, bool rotation_invariant,
                      bool scale_invariant, float pattern_scale);

  /// \brief Initialize the freak pipeline with a preprocessing pipeline.
  ///
  /// \param[in] preprocessing      An undistorter to do preprocessing such as
  ///                               contrast enhancement or undistortion.
  /// \param[in] copy_images        Should we deep copy the images passed in?
  /// \param[in] octaves            Number of octaves for brisk/freak scale computation.
  /// \param[in] uniformity_radius  Gets multiplied on the keypoint scale and determines
  ///                               the distance to the next neighboring keypoint.
  /// \param[in] absolute_threshold The harris absolute threshold.
  ///                               Low makes more keypoints.
  /// \param[in] max_number_of_keypoints The maximum number of keypoints to return.
  /// \param[in] rotation_invariant Should freak estimate the keypoint orientation?
  /// \param[in] scale_invariant    Should freak estimate the keypoint scale?
  /// \param[in] pattern_scale Scale of the pattern for the freak feature descriptor.
  FreakVisualPipeline(std::unique_ptr<Undistorter>& preprocessing, bool copy_images, size_t octaves,
                      double uniformity_radius, double absolute_threshold,
                      size_t max_number_of_keypoints, bool rotation_invariant,
                      bool scale_invariant, float pattern_scale);

  virtual ~FreakVisualPipeline();

  /// \brief Initialize the freak pipeline.
  ///
  /// \param[in] octaves            Number of octaves for FREAK scale computation.
  /// \param[in] uniformity_radius  Gets multiplied on the keypoint scale and determines
  ///                               the distance to the next neighboring keypoint.
  /// \param[in] absolute_threshold The harris absolute threshold.
  ///                               Low makes more keypoints.
  /// \param[in] max_number_of_keypoints The maximum number of keypoints to return.
  /// \param[in] rotation_invariant Should freak estimate the keypoint orientation?
  /// \param[in] scale_invariant    Should freak estimate the keypoint scale?
  /// \param[in] pattern_scale Scale of the pattern for the freak feature descriptor.
  void initializeFreak(size_t octaves, double uniformity_radius,
                       double absolute_threshold, size_t max_number_of_keypoints,
                       bool rotation_invariant, bool scale_invariant, float pattern_scale);

protected:
  /// \brief Process the frame and fill the results into the frame variable
  ///
  /// The top level function will already fill in the timestamps and the output camera.
  /// \param[in]     image The image data.
  /// \param[in/out] frame The visual frame. This will be constructed before calling.
  virtual void processFrameImpl(const cv::Mat& image,
                                VisualFrame* frame) const;
private:
  std::shared_ptr<cv::FeatureDetector> detector_;
  std::shared_ptr<cv::DescriptorExtractor> extractor_;

  size_t octaves_;
  double uniformity_radius_;
  double absolute_threshold_;
  size_t max_number_of_keypoints_;
  bool rotation_invariant_;
  bool scale_invariant_;
  float pattern_scale_;
};

}  // namespace aslam

#endif // ASLAM_FREAK_PIPELINE_H_
