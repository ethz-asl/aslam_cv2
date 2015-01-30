#include <map>
#include <memory>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>

#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/feature-tracker.h>

namespace aslam { class Camera; }

namespace aslam {
FeatureTracker::FeatureTracker()
    : camera_(nullptr),
      current_track_id_(0) {}

FeatureTracker::FeatureTracker(const aslam::Camera::ConstPtr& input_camera)
    : camera_(input_camera),
      current_track_id_(0) {
  CHECK(camera_);
}

void FeatureTracker::drawTracks(const VisualFrame& current_frame, cv::Mat* track_image) {
  CHECK_NOTNULL(track_image);

  // Create the image
  const cv::Mat& raw_image = current_frame.getRawImage();
  cv::cvtColor(raw_image, *track_image, CV_GRAY2BGR);

  // Check for new tracks and store it.
  Eigen::VectorXi current_track_ids = current_frame.getTrackIds();
  std::map<int, Track> current_tracks;

  // Find currently active tracks
  const size_t num_keypoints = current_frame.getNumKeypointMeasurements();
  CHECK_EQ(current_track_ids.rows(), static_cast<int>(num_keypoints));
  for (size_t i = 0; i < num_keypoints; i++) {
    const int track_id = current_track_ids(i);
    if (track_id != -1) {
      Track track;
      track.track_id = track_id;
      current_frame.toRawImageCoordinates(current_frame.getKeypointMeasurement(i),
                                          &track.end_point);

      if (previous_drawn_tracks_.count(track_id) > 0) {
        // Continued track.
        track.starting_point = previous_drawn_tracks_[track_id].starting_point;
        track.length = previous_drawn_tracks_[track_id].length + 1;
      } else {
        // New track.
        track.starting_point = track.end_point;
        track.length = 1;
      }
      current_tracks.insert(std::pair<int, Track>(track_id, track));
    }
  }

  // Draw the tracks
  auto drawTrack = [&track_image](const Track& track) {
    cv::circle(*track_image, cv::Point(track.end_point[0], track.end_point[1]),
        4, CV_RGB(0, 180, 180));
    cv::line(*track_image, cv::Point(track.starting_point[0], track.starting_point[1]),
        cv::Point(track.end_point[0], track.end_point[1]),
        CV_RGB(110, 255, 110));
    cv::putText(*track_image, std::to_string(track.track_id) + " / " + std::to_string(track.length),
        cv::Point(track.end_point[0], track.end_point[1]),
        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255));
  };

  for (auto current_track : current_tracks) {
    drawTrack(current_track.second);
  }

  // Swap the current and previous
  previous_drawn_tracks_.swap(current_tracks);
}

}  // namespace aslam
