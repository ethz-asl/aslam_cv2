#include "aslam/visualization/feature-track-visualizer.h"

#include <glog/logging.h>

namespace aslam_cv_visualization {

void FeatureTrackVisualizer::drawContinuousFeatureTracks(
    aslam::VisualFrame::ConstPtr frame,
    const aslam::FeatureTracks& terminated_feature_tracks,
    cv::Mat* image) {
  if (last_frame_) {
    VLOG(4) << "Last frame hast ID " << last_frame_->getId();
    // Preprocess the last frame. It has changed in the meantime.
    typedef aslam::AlignedUnorderedMap<int, size_t>::type TrackIdToIndexMap;
    TrackIdToIndexMap last_frame_track_id_to_index_map;

    const Eigen::VectorXi& last_frame_track_ids = last_frame_->getTrackIds();
    const size_t last_frame_num_track_ids =
        static_cast<size_t>(last_frame_track_ids.rows());
    CHECK_EQ(last_frame_num_track_ids,
             last_frame_->getNumKeypointMeasurements());
    for (size_t idx = 0; idx < last_frame_num_track_ids; ++idx) {
      int track_id = last_frame_track_ids(idx);
      if (track_id >= 0) {
        last_frame_track_id_to_index_map.insert(std::make_pair(track_id, idx));
      }
    }

    VLOG(4) << "Pre-processed " << last_frame_track_id_to_index_map.size()
        << " valid keypoints in last frame.";

    const Eigen::VectorXi& track_ids = frame->getTrackIds();

    const size_t num_track_ids = static_cast<size_t>(track_ids.rows());
    CHECK_EQ(num_track_ids, frame->getNumKeypointMeasurements());
    for (size_t idx = 0; idx < num_track_ids; ++idx) {
      int track_id = track_ids(idx);
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
          Eigen::Vector2d measurement =
              last_frame_->getKeypointMeasurement(last_frame_it->second);
          new_track.keypoints.push_back(measurement);
          track_id_to_track_map_.insert(std::make_pair(track_id, new_track));
          active_track_it = track_id_to_track_map_.find(track_id);
        }
        CHECK(active_track_it != track_id_to_track_map_.end());

        Eigen::Vector2d measurement = frame->getKeypointMeasurement(idx);
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
      CHECK_GT(track_it->second.keypoints.size(), 0);
      bool terminated = (terminated_track_ids.count(track_it->first) > 0u);
      aslam::Aligned<std::vector, Eigen::Vector2d>::type::const_iterator
        measurement_it = track_it->second.keypoints.begin();
      cv::Point p0((*measurement_it)(0), (*measurement_it)(1));
      ++measurement_it;
      for (; measurement_it != track_it->second.keypoints.end();
          ++measurement_it) {
        cv::Point p1((*measurement_it)(0), (*measurement_it)(1));
        if (terminated) {
          cv::line(*image, p0, p1, cv::Scalar(0, 0, 255), 1, CV_AA);
        } else {
          cv::line(*image, p0, p1, track_it->second.color, 1, CV_AA);
        }
        p0 = p1;
      }
      if (!terminated) {
        cv::circle(*image, p0, 1, cv::Scalar(0.0, 255.0, 255.0, 20), 1, CV_AA);
      }
    }

    for (const aslam::FeatureTrack& track : terminated_feature_tracks) {
      int track_id = static_cast<int>(track.getTrackId());
      TrackMap::iterator it = track_id_to_track_map_.find(track_id);
      CHECK(it != track_id_to_track_map_.end());
      track_id_to_track_map_.erase(track_id);
    }
  } else {
    VLOG(4) << "First time in visualizer. Asssining last frame and returning.";
  }
  last_frame_ = frame;
}
}  // namespace aslam_cv_visualization
