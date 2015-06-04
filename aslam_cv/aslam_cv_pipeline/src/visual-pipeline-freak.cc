#include <aslam/pipeline/visual-pipeline-freak.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/pipeline/undistorter.h>
#include <brisk/brisk.h>
#include <glog/logging.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

namespace aslam {

FreakVisualPipeline::FreakVisualPipeline() {
  // Just for serialization. Not meant to be used.
}

FreakVisualPipeline::FreakVisualPipeline(const Camera::ConstPtr& camera,
                                         bool copy_images,
                                         size_t num_octaves,
                                         int hessian_threshold,
                                         int num_octave_layers,
                                         bool rotation_invariant,
                                         bool scale_invariant,
                                         float pattern_scale)
: VisualPipeline(camera, camera, copy_images) {
  if (cv::initModule_nonfree()) {
    initializeFreak(num_octaves, hessian_threshold, num_octave_layers,
                    rotation_invariant, scale_invariant, pattern_scale);
  } else {
    LOG(ERROR) << "Could not initialize opencv nonfree module.";
  }
}

FreakVisualPipeline::FreakVisualPipeline(
                                   std::unique_ptr<Undistorter>& preprocessing,
                                         bool copy_images,
                                         size_t num_octaves,
                                         int hessian_threshold,
                                         int num_octave_layers,
                                         bool rotation_invariant,
                                         bool scale_invariant,
                                         float pattern_scale)
: VisualPipeline(preprocessing, copy_images) {
  if (cv::initModule_nonfree()) {
    initializeFreak(num_octaves, hessian_threshold, num_octave_layers,
                    rotation_invariant, scale_invariant, pattern_scale);
  } else {
    LOG(ERROR) << "Could not initialize opencv nonfree module.";
  }
}

FreakVisualPipeline::~FreakVisualPipeline() { }

void FreakVisualPipeline::initializeFreak(size_t num_octaves,
                                          int hessian_threshold,
                                          int num_octave_layers,
                                          bool rotation_invariant,
                                          bool scale_invariant,
                                          float pattern_scale) {
  hessian_threshold_ = hessian_threshold;
  octaves_ = num_octaves;
  num_octave_layers_ =  num_octave_layers;
  rotation_invariant_ = rotation_invariant;
  scale_invariant_ = scale_invariant;
  pattern_scale_ = pattern_scale;

  detector_.reset(new cv::SurfFeatureDetector(octaves_,
                                              hessian_threshold_,num_octave_layers_));
  extractor_.reset(new cv::FREAK(rotation_invariant_,
                                 scale_invariant_, pattern_scale_, octaves_));
}

void FreakVisualPipeline::processFrameImpl(const cv::Mat& image, VisualFrame* frame) const {
  CHECK_NOTNULL(frame);
  // Now we use the image from the frame. It might be undistorted.
  std::vector<cv::KeyPoint> keypoints;
  detector_->detect(image, keypoints);

  cv::Mat descriptors;
  if(!keypoints.empty()) {
    extractor_->compute(image, keypoints, descriptors);
  } else {
    descriptors = cv::Mat(0, 0, CV_8UC1);
    LOG(WARNING) << "Frame produced no keypoints:\n" << *frame;
  }
  // Note: It is important that
  //       (a) this happens after the descriptor extractor as the extractor
  //           may remove keypoints; and
  //       (b) the values are set even if there are no keypoints as downstream
  //           code may rely on the keypoints being set.
  CHECK_EQ(descriptors.type(), CV_8UC1);
  CHECK(descriptors.isContinuous());
  frame->setDescriptors(
      // Switch cols/rows as Eigen is col-major and cv::Mat is row-major
      Eigen::Map<VisualFrame::DescriptorsT>(descriptors.data,
                                            descriptors.cols,
                                            descriptors.rows)
  );

  // The keypoint uncertainty is set to a constant value.
  const double kKeypointUncertaintyPixelSigma = 0.8;

  Eigen::Matrix2Xd ikeypoints(2, keypoints.size());
  Eigen::VectorXd scales(keypoints.size());
  Eigen::VectorXd orientations(keypoints.size());
  Eigen::VectorXd scores(keypoints.size());
  Eigen::VectorXd uncertainties(keypoints.size());

  // \TODO(ptf) Who knows a good formula for uncertainty based on octave?
  //            See https://github.com/ethz-asl/aslam_cv2/issues/73
  for(size_t i = 0; i < keypoints.size(); ++i) {
    const cv::KeyPoint& kp = keypoints[i];
    ikeypoints(0,i)  = kp.pt.x;
    ikeypoints(1,i)  = kp.pt.y;
    scales[i]        = kp.size;
    orientations[i]  = kp.angle;
    scores[i]        = kp.response;
    uncertainties[i] = kKeypointUncertaintyPixelSigma;
  }
  frame->swapKeypointMeasurements(&ikeypoints);
  frame->swapKeypointScores(&scores);
  frame->swapKeypointOrientations(&orientations);
  frame->swapKeypointScales(&scales);
  frame->swapKeypointMeasurementUncertainties(&uncertainties);
}

}  // namespace aslam
