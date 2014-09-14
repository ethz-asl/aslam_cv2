#include <aslam/pipeline/visual-pipeline-brisk.h>

#include <aslam/frames/visual-frame.h>
#include <aslam/pipeline/undistorter.h>
#include <brisk/brisk.h>
#include <glog/logging.h>

namespace aslam {

BriskVisualPipeline::BriskVisualPipeline() {
  // Just for serialization. Not meant to be used.
}

BriskVisualPipeline::BriskVisualPipeline(
    const std::shared_ptr<Camera>& camera, bool copy_images, size_t octaves,
    double uniformity_radius, double absolute_threshold,
    size_t max_number_of_keypoints, bool rotation_invariant,
    bool scale_invariant) : VisualPipeline(camera, camera, copy_images) {
  CHECK_NOTNULL(camera.get());
  initializeBrisk(octaves, uniformity_radius, absolute_threshold,
                  max_number_of_keypoints, rotation_invariant,
                  scale_invariant);
}

BriskVisualPipeline::BriskVisualPipeline(
    const Undistorter::Ptr& preprocessing, bool copy_images, size_t octaves,
    double uniformity_radius, double absolute_threshold,
    size_t max_number_of_keypoints, bool rotation_invariant,
    bool scale_invariant) :
        VisualPipeline(preprocessing, copy_images) {
  initializeBrisk(octaves, uniformity_radius, absolute_threshold,
                  max_number_of_keypoints, rotation_invariant,
                  scale_invariant);
}

BriskVisualPipeline::~BriskVisualPipeline() { }

void BriskVisualPipeline::initializeBrisk(size_t octaves,
                                          double uniformity_radius,
                                          double absolute_threshold,
                                          size_t max_number_of_keypoints,
                                          bool rotation_invariant,
                                          bool scale_invariant) {
  octaves_ = octaves;
  uniformity_radius_ = uniformity_radius;
  absolute_threshold_ = absolute_threshold;
  max_number_of_keypoints_ = max_number_of_keypoints;
  rotation_invariant_ = rotation_invariant;
  scale_invariant_ = scale_invariant;
#if __arm__
  // \TODO(slynen): Currently no Harris on ARM. Adapt if we port it to ARM.
  static const int AstThreshold = 70;
  detector_.reset(new brisk::BriskFeatureDetector(AstThreshold) );
#else
  detector_.reset(
      new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
          octaves_, uniformity_radius_, absolute_threshold_, max_number_of_keypoints_)
  );
#endif
  extractor_.reset(new brisk::BriskDescriptorExtractor(rotation_invariant_,
                                                       scale_invariant_));
}

void BriskVisualPipeline::processFrameImpl(const cv::Mat& image,
                                           VisualFrame* frame) const {
  CHECK_NOTNULL(frame);
  // Now we use the image from the frame. It might be undistorted.
  std::vector <cv::KeyPoint> keypoints;
  detector_->detect(image, keypoints);

  if(!keypoints.empty()) {
    // Copy over the keypoints
    // \TODO(slynen)  can you think of a way to avoid the copy when setting
    //                the data in the channels? Some allocate method to get
    //                a reference to the channel data directly?
    Eigen::Matrix2Xd ikeypoints(2, keypoints.size());
    Eigen::VectorXd  scales(keypoints.size());
    Eigen::VectorXd  orientations(keypoints.size());
    // \TODO(ptf) Who knows a good formula for uncertainty based on octave?
    for(size_t i = 0; i < keypoints.size(); ++i) {
      const cv::KeyPoint& kp = keypoints[i];
      ikeypoints(0,i)  = kp.pt.x;
      ikeypoints(1,i)  = kp.pt.y;
      scales[i]        = kp.size;
      orientations[i]  = kp.angle;
    }
    frame->setKeypointMeasurements(ikeypoints);
    frame->setKeypointOrientations(orientations);
    frame->setKeypointScales(scales);

    // Make the ocv descriptor matrix point to the eigen matrix we will use for descriptors.
    cv::Mat descriptors;
    extractor_->compute(image, keypoints, descriptors);
    CHECK_EQ(descriptors.type(), CV_8UC1);
    CHECK(descriptors.isContinuous());
    frame->setBriskDescriptors(
        Eigen::Map<VisualFrame::DescriptorsT>(descriptors.data,
                                              descriptors.rows,
                                              descriptors.cols)
    );
  } else {
    LOG(WARNING) << "Frame produced no keypoints:\n" << *frame;
  }
}


}  // namespace aslam
