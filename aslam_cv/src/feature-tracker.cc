#include <memory>

#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>

#include <aslam/tracker/feature-tracker.h>
#include <aslam/frames/visual-frame.h>

// Some forward declarations.
namespace aslam { class Camera; }

namespace aslam {
FeatureTracker::FeatureTracker()
    : camera_(nullptr),
      current_track_id_(0) {}

FeatureTracker::FeatureTracker(const std::shared_ptr<const aslam::Camera>& input_camera)
    : camera_(CHECK_NOTNULL(input_camera.get())),
      current_track_id_(0) {}

void FeatureTracker::drawTracks(std::shared_ptr<VisualFrame> current_frame_ptr,
                                cv::Mat* track_image) {

  CHECK_NOTNULL(current_frame_ptr.get());
  CHECK_NOTNULL(track_image);

  struct Track {
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW             // TODO(schneith): Do we need this?
    Track() : track_id(-1), length(0) {};
    int track_id;
    Eigen::Vector2d starting_point;
    Eigen::Vector2d end_point;
    int length;
  };

  static std::map<int, Track> previous_tracks;
  const VisualFrame& current_frame = *current_frame_ptr;

  // Create the image
  const cv::Mat& raw_image = current_frame.getRawImage();
  cv::cvtColor(raw_image, *track_image, CV_GRAY2BGR);

  // Check for new tracks and store it.
  Eigen::VectorXi current_track_ids = current_frame.getTrackIds();
  std::map<int, Track> current_tracks;

  // Find currently active tracks
  CHECK_EQ(current_track_ids.rows(), current_frame.getKeypointMeasurements().cols());
  for (int i = 0; i < current_frame.getTrackIds().rows(); i++) {
    const int track_id = current_track_ids(i);
    if (track_id != -1) {
      CHECK_LT(i, current_frame.getKeypointMeasurements().cols());

      Track track;
      track.track_id = track_id;
      track.end_point = current_frame.getKeypointMeasurement(i);

      if (previous_tracks.count(track_id) > 0) {
        // Continued track.
        track.starting_point = previous_tracks[track_id].starting_point;
        track.length = previous_tracks[track_id].length + 1;
      } else {
        // New track.
        track.starting_point = track.end_point;
      }
      current_tracks.insert(std::pair<int, Track>(track_id, track));
    }
  }

  // Draw the tracks
  auto drawTrack = [track_image](const Track& track) {
    cv::circle(*track_image, cv::Point(track.end_point[0], track.end_point[1]),
        4, CV_RGB(0, 180, 180));
    cv::line(*track_image, cv::Point(track.starting_point[0], track.starting_point[1]),
        cv::Point(track.end_point[0], track.end_point[1]),
        CV_RGB(110, 255, 110));
    cv::putText(*track_image, std::to_string(track.track_id),
        cv::Point(track.end_point[0], track.end_point[1]),
        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));
  };

  for (auto current_track : current_tracks) {
    drawTrack(current_track.second);
  }

  // Swap the current and previous
  previous_tracks.swap(current_tracks);
}

}  // namespace aslam
