#ifndef ASLAM_VISUALIZATION_BASIC_VISUALIZATION_H_
#define ASLAM_VISUALIZATION_BASIC_VISUALIZATION_H_

#include <vector>

#include <aslam/common/memory.h>
#include <aslam/frames/feature-track.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <aslam/matcher/match.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace aslam_cv_visualization {

// Colors are in BGR8.
const cv::Scalar kBlue(255, 0, 0);
const cv::Scalar kGreen(0, 255, 0);
const cv::Scalar kBrightGreen(110, 255, 110);
const cv::Scalar kRed(0, 0, 255);
const cv::Scalar kYellow(0, 255, 255);
const cv::Scalar kTurquoise(180, 180, 0);
const cv::Scalar kBlack(0, 0, 0);
const cv::Scalar kWhite(255, 255, 255);

struct ImagePositionOffset {
  size_t width;
  size_t height;
};
typedef std::vector<ImagePositionOffset> Offsets;

////////////////////////////////////////////////
/// High-level functions - They render raw images plus some additional visualization.
////////////////////////////////////////////////

/// Takes an nframe, assembles all frames into one big image, puts in the raw images from the
/// frames and draws keypoints on them.
void visualizeKeypoints(const std::shared_ptr<const aslam::VisualNFrame>& nframe, cv::Mat* image);

/// Takes two frames and list of matches between them and draws the raw images and the matches.
template<typename MatchesWithScore>
void visualizeMatches(const aslam::VisualFrame& frame_kp1,
                      const aslam::VisualFrame& frame_k,
                      const MatchesWithScore& matches,
                      cv::Mat* image);

////////////////////////////////////////////////
/// Low-Level functions - They only draw some visualization onto an image.
////////////////////////////////////////////////

/// Takes a frame and draws keypoints on it.
void drawKeypoints(const aslam::VisualFrame& frame, cv::Mat* image);

/// Takes two frames and a list of matches between them and draws the matches.
/// Does not draw the raw image!
void drawKeypointMatches(const aslam::VisualFrame& frame_kp1,
                         const aslam::VisualFrame& frame_k,
                         const aslam::Matches& matches_kp1_k,
                         cv::Scalar color_keypoint_kp1, cv::Scalar line_color,
                         cv::Mat* image);

/// Takes an nframe and creates a single image patching together all raw images of all frames.
void assembleMultiImage(const aslam::VisualNFrame::ConstPtr& nframe,
                        cv::Mat* full_image, Offsets* offsets);

/// Draw the patches around all keypoints for one features tracks.
void drawFeatureTrackPatches(const aslam::FeatureTrack& track, size_t neighborhood_px,
                             cv::Mat* image);

/// Draw the patches around all keypoints for a list of features tracks.
bool drawFeatureTracksPatches(const aslam::FeatureTracks& tracks, size_t neighborhood_px,
                              size_t num_cols, cv::Mat* all_tracks_image);

/// Draw a list of feature tracks into an image. The tracks are drawn into the image
/// associated with the last keypoint of the first track in the list.
bool drawFeatureTracks(const aslam::FeatureTracks& tracks, cv::Mat* image);

}  // namespace aslam_cv_visualization

#include "./basic-visualization-inl.h"

#endif  // ASLAM_VISUALIZATION_BASIC_VISUALIZATION_H_
