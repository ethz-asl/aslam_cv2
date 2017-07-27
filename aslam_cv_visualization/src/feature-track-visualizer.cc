#include "aslam/visualization/feature-track-visualizer.h"

#include <glog/logging.h>

#include "aslam/visualization/basic-visualization.h"

namespace aslam_cv_visualization {

const size_t kLineWidth = 1u;
const size_t kCircleRadius = 1u;

void VisualFrameFeatureTrackVisualizer::drawContinuousFeatureTracks(
    const aslam::VisualFrame::ConstPtr& frame,
    const aslam::FeatureTracks& terminated_feature_tracks,
    cv::Mat* image) {
  CHECK_NOTNULL(image);

  if (last_frame_) {
    VLOG(4) << "Last frame has ID " << last_frame_->getId();

    // Preprocess the last frame. It has changed in the meantime.
    TrackIdToIndexMap last_frame_track_id_to_index_map;
    preprocessLastFrame(&last_frame_track_id_to_index_map);

    VLOG(4) << "Pre-processed " << last_frame_track_id_to_index_map.size()
            << " valid keypoints in last frame.";

    const Eigen::VectorXi& track_ids = frame->getTrackIds();

    const size_t num_track_ids = static_cast<size_t>(track_ids.rows());
    CHECK_EQ(num_track_ids, frame->getNumKeypointMeasurements());
    for (size_t idx = 0; idx < num_track_ids; ++idx) {
      const int track_id = track_ids(idx);
      if (track_id >= 0) {
        // Active track. Look if this is ongoing.
        TrackMap::iterator active_track_it =
            track_id_to_track_map_.find(track_id);
        if (active_track_it == track_id_to_track_map_.end()) {
          // This is a new track. Get the first two keypoints using the last frame.
          TrackIdToIndexMap::iterator last_frame_it =
              last_frame_track_id_to_index_map.find(track_id);
          CHECK(last_frame_it != last_frame_track_id_to_index_map.end())
              << "Could not find track with ID "
              << track_id << " in the last frame, although this is a new track. "
              << "Current keypoint index: " << idx;
          CHECK_LT(last_frame_it->second,
                   last_frame_->getNumKeypointMeasurements());

          Track new_track;
          new_track.color = cv::Scalar(rng_.uniform(0, 255),
                                       rng_.uniform(0, 255),
                                       rng_.uniform(0, 255));
          const Eigen::Vector2d measurement =
              last_frame_->getKeypointMeasurement(last_frame_it->second);
          new_track.keypoints.push_back(measurement);
          track_id_to_track_map_.insert(std::make_pair(track_id, new_track));
          active_track_it = track_id_to_track_map_.find(track_id);
        }
        CHECK(active_track_it != track_id_to_track_map_.end());

        const Eigen::Vector2d measurement = frame->getKeypointMeasurement(idx);
        active_track_it->second.keypoints.push_back(measurement);
      }
    }

    std::unordered_set<int> terminated_track_ids;
    for (const aslam::FeatureTrack& track : terminated_feature_tracks) {
      int track_id = static_cast<int>(track.getTrackId());
      terminated_track_ids.insert(track_id);
    }

    for (TrackMap::const_iterator track_it = track_id_to_track_map_.begin();
        track_it != track_id_to_track_map_.end(); ++track_it) {
      CHECK_GT(track_it->second.keypoints.size(), 0u);
      bool terminated = (terminated_track_ids.count(track_it->first) > 0u);
      Aligned<std::vector, Eigen::Vector2d>::const_iterator
        measurement_it = track_it->second.keypoints.begin();
      cv::Point point_start_line((*measurement_it)(0), (*measurement_it)(1));
      ++measurement_it;
      for (; measurement_it != track_it->second.keypoints.end();
          ++measurement_it) {
        cv::Point point_end_line((*measurement_it)(0), (*measurement_it)(1));
        if (terminated) {
          cv::line(*image, point_start_line, point_end_line, cv::Scalar(0, 0, 255), kLineWidth,
                   CV_AA);
        } else {
          cv::line(*image, point_start_line, point_end_line, track_it->second.color, kLineWidth,
                   CV_AA);
        }
        point_start_line = point_end_line;
      }
      if (!terminated) {
        cv::circle(*image, point_start_line, kCircleRadius, cv::Scalar(0.0, 255.0, 255.0, 20),
                   kLineWidth, CV_AA);
      }
    }

    for (const aslam::FeatureTrack& track : terminated_feature_tracks) {
      int track_id = static_cast<int>(track.getTrackId());
      TrackMap::iterator it = track_id_to_track_map_.find(track_id);
      CHECK(it != track_id_to_track_map_.end());
      track_id_to_track_map_.erase(track_id);
    }
  }
  last_frame_ = frame;
}

void VisualFrameFeatureTrackVisualizer::preprocessLastFrame(
    TrackIdToIndexMap* last_frame_track_id_to_index_map) {
  CHECK_NOTNULL(last_frame_track_id_to_index_map);
  CHECK(last_frame_) << "No last frame available.";

  const Eigen::VectorXi& last_frame_track_ids = last_frame_->getTrackIds();
  const size_t last_frame_num_track_ids =
      static_cast<size_t>(last_frame_track_ids.rows());
  CHECK_EQ(last_frame_num_track_ids,
           last_frame_->getNumKeypointMeasurements());
  for (size_t idx = 0; idx < last_frame_num_track_ids; ++idx) {
    int track_id = last_frame_track_ids(idx);
    if (track_id >= 0) {
      last_frame_track_id_to_index_map->insert(std::make_pair(track_id, idx));
    }
  }
}

VisualNFrameFeatureTrackVisualizer::VisualNFrameFeatureTrackVisualizer(const size_t num_frames) {
  CHECK_GT(num_frames, 0u);
  feature_track_visualizers_.resize(num_frames);
}

void VisualNFrameFeatureTrackVisualizer::drawContinuousFeatureTracks(
    const aslam::VisualNFrame::ConstPtr& nframe,
    const aslam::FeatureTracksList& terminated_feature_tracks,
    cv::Mat* image) {
  CHECK(nframe);
  CHECK_NOTNULL(image);

  const size_t num_frames = nframe->getNumFrames();
  CHECK_EQ(num_frames, feature_track_visualizers_.size());
  CHECK_EQ(nframe->getNumCameras(), num_frames);

  cv::Mat& full_image = *image;
  Offsets offsets;
  assembleMultiImage(nframe, &full_image, &offsets);
  CHECK_EQ(offsets.size(), num_frames);

  for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
    const size_t image_width = nframe->getCamera(frame_idx).imageWidth();
    const size_t image_height = nframe->getCamera(frame_idx).imageHeight();

    cv::Mat slice = full_image(cv::Rect(offsets[frame_idx].width, offsets[frame_idx].height,
                                        image_width, image_height));

    feature_track_visualizers_[frame_idx].drawContinuousFeatureTracks(
        nframe->getFrameShared(frame_idx), terminated_feature_tracks[frame_idx], &slice);
  }
}

void VisualNFrameFeatureTrackVisualizer::setNumFrames(const size_t num_frames) {
  CHECK_GT(num_frames, 0u);
  feature_track_visualizers_.resize(num_frames);
}

}  // namespace aslam_cv_visualization
