#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-landmarks-to-frame.h"

namespace aslam {

MatchingProblemLandmarksToFrame::MatchingProblemLandmarksToFrame(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList& landmarks,
    double image_space_distance_threshold_pixels,
    int hamming_distance_threshold)
  : landmarks_(landmarks), frame_(frame),
    descriptor_size_bytes_(frame.getDescriptorSizeBytes()),
    descriptor_size_bits_(static_cast<int>(frame.getDescriptorSizeBytes() * 8u)),
    squared_image_space_distance_threshold_pixels_squared_(
        image_space_distance_threshold_pixels * image_space_distance_threshold_pixels),
    hamming_distance_threshold_(hamming_distance_threshold) {
  CHECK_GT(hamming_distance_threshold, 0) << "Descriptor distance needs to be positive.";
  CHECK_GT(image_space_distance_threshold_pixels, 0.0)
    << "Image space distance needs to be positive.";
  CHECK(frame.getCameraGeometry()) << "The camera of the visual frame is NULL.";

  image_height_frame_ = frame.getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_frame_, 0u) << "The visual frame has zero image rows.";
  CHECK_GT(descriptor_size_bytes_, 0);
  CHECK_GT(descriptor_size_bits_, 0);
}

size_t MatchingProblemLandmarksToFrame::numApples() const {
  return frame_.getNumKeypointMeasurements();
}

size_t MatchingProblemLandmarksToFrame::numBananas() const {
  return landmarks_.size();
}

}  // namespace aslam
