#ifndef VISUALIZATION_FEATURE_TRACK_VISUALIZER_H_
#define VISUALIZATION_FEATURE_TRACK_VISUALIZER_H_

#include <aslam/common/memory.h>
#include <aslam/frames/feature-track.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

namespace aslam_cv_visualization {

/// Visualization for aslam::FeatureTracks.
class VisualFrameFeatureTrackVisualizer {
 public:
  VisualFrameFeatureTrackVisualizer() : rng_(cv::RNG(0xFFFFFFFF)) {}

  void drawContinuousFeatureTracks(
      const aslam::VisualFrame::ConstPtr& frame,
      const aslam::FeatureTracks& terminated_feature_tracks,
      cv::Mat* image);

 private:
  typedef AlignedUnorderedMap<int, size_t> TrackIdToIndexMap;

  void preprocessLastFrame(TrackIdToIndexMap* track_id_to_keypoint_index_map);

  struct Track {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Aligned<std::vector, Eigen::Vector2d> keypoints;
    cv::Scalar color;
  };
  typedef AlignedUnorderedMap<int, Track> TrackMap;
  TrackMap track_id_to_track_map_;

  aslam::VisualFrame::ConstPtr last_frame_;

  cv::RNG rng_;
};

class VisualNFrameFeatureTrackVisualizer {
 public:
  VisualNFrameFeatureTrackVisualizer() = default;
  explicit VisualNFrameFeatureTrackVisualizer(const size_t num_frames);

  void drawContinuousFeatureTracks(
      const aslam::VisualNFrame::ConstPtr& nframe,
      const aslam::FeatureTracksList& terminated_feature_tracks,
      cv::Mat* image);

  void setNumFrames(const size_t num_frames);

 private:
  std::vector<VisualFrameFeatureTrackVisualizer> feature_track_visualizers_;
};
}  // namespace aslam_cv_visualization
#endif  // VISUALIZATION_FEATURE_TRACK_VISUALIZER_H_
