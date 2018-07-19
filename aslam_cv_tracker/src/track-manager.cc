#include <set>

#include <glog/logging.h>

#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/match.h>

#include "aslam/tracker/track-manager.h"

namespace aslam {
  ThreadSafeIdProvider<size_t> TrackManager::track_id_provider_(0u);

  Eigen::VectorXi* TrackManager::createAndGetTrackIdChannel(VisualFrame* frame) {
    // Load (and create) track id channels.
    CHECK_NOTNULL(frame);
    size_t num_track_ids = frame->getNumKeypointMeasurements();
    if (!frame->hasTrackIds()) {
      frame->setTrackIds(Eigen::VectorXi::Constant(num_track_ids, -1));
    }
    return CHECK_NOTNULL(frame->getTrackIdsMutable());
  }

  void TrackManager::applyStereoMatchesToFrames(
      const StereoMatchesWithScore& stereo_matches, VisualFrame* frame0,
      VisualFrame* frame1) {
    CHECK_NOTNULL(apple_frame);
    CHECK_NOTNULL(banana_frame);

    // Get the track ID channels.
    Eigen::VectorXi& frame0_track_ids = *CHECK_NOTNULL(createAndGetTrackIdChannel(frame0));
    Eigen::VectorXi& frame1_track_ids = *CHECK_NOTNULL(createAndGetTrackIdChannel(frame1));

    size_t num_frame0_track_ids = static_cast<size_t>(frame0_track_ids.rows());
    size_t num_frame1_track_ids = static_cast<size_t>(frame1_track_ids.rows());

    std::unordered_set<int> used_frame0_ids;
    std::unordered_set<int> used_frame1_ids;

    for (const StereoMatchWithScore& match : matches_A_B) {
      int index_frame0 = match.getKeypointIndexFrame0();
      CHECK_LT(index_frame0, static_cast<int>(num_frame0_track_ids));
      CHECK_GE(index_frame0, 0);

      int index_frame1 = match.getKeypointIndexFrame1();
      CHECK_LT(index_frame1, static_cast<int>(num_frame1_track_ids));
      CHECK_GE(index_frame1, 0);

      addToSetsAndCheckExclusiveness(index_frame0,
                                     index_frame1,
                                     &used_frame0_ids,
                                     &used_frame1_ids);

      int track_id_frame0 = frame0_track_ids(index_frame0);
      int track_id_frame1 = frame1_track_ids(index_frame0);
      
      // Fuse feature tracks if possible.
      if (track_id_frame0 == track_id_frame1) {
        // Tracks are already merged.
        continue;
      }
      if ((track_id_frame0) < 0 || (track_id_frame1 < 0)) {
        // One of the keypoints is not tracked, omitting the match.
        continue;
      } 
    }
  }
  void SimpleTrackManager::applyMatchesToFrames(
      const FrameToFrameMatchesWithScore& matches_A_B, VisualFrame* apple_frame,
      VisualFrame* banana_frame) {
    CHECK_NOTNULL(apple_frame);
    CHECK_NOTNULL(banana_frame);

    // Get the track ID channels.
    Eigen::VectorXi& apple_track_ids = *CHECK_NOTNULL(createAndGetTrackIdChannel(apple_frame));
    Eigen::VectorXi& banana_track_ids = *CHECK_NOTNULL(createAndGetTrackIdChannel(banana_frame));

    size_t num_apple_track_ids = static_cast<size_t>(apple_track_ids.rows());
    size_t num_banana_track_ids = static_cast<size_t>(banana_track_ids.rows());

    std::unordered_set<int> consumed_apples;
    std::unordered_set<int> consumed_bananas;

    for (const FrameToFrameMatchWithScore& match : matches_A_B) {
      int index_apple = match.getKeypointIndexAppleFrame();
      CHECK_LT(index_apple, static_cast<int>(num_apple_track_ids));
      CHECK_GE(index_apple, 0);

      int index_banana = match.getKeypointIndexBananaFrame();
      CHECK_LT(index_banana, static_cast<int>(num_banana_track_ids));
      CHECK_GE(index_banana, 0);

      addToSetsAndCheckExclusiveness(index_apple,
                                     index_banana,
                                     &consumed_apples,
                                     &consumed_bananas);

      int track_id_apple = apple_track_ids(index_apple);
      int track_id_banana = banana_track_ids(index_banana);

      if ((track_id_apple) < 0 && (track_id_banana < 0)) {
        // Both track ids are < 0. Start a new track.
        int new_track_id = track_id_provider_.getNewId();
        apple_track_ids(index_apple) = new_track_id;
        banana_track_ids(index_banana) = new_track_id;
      } else {
        // Either one of the track ids is >= 0.
        if (track_id_apple != track_id_banana) {
          // The two track ids are not equal, so we need to copy one over to the
          // other.
          if (track_id_banana >= 0) {
            CHECK_LT(track_id_apple, 0) << "Both the apple and the banana track "
                "id are >= 0 but they are not equal!";

            apple_track_ids(index_apple) = track_id_banana;
          } else {
            CHECK_LT(track_id_banana, 0) << "Both the apple and the banana "
                "track id are >= 0 but they are not equal!";

            banana_track_ids(index_banana) = track_id_apple;
          }
        }
      }
    }
  }

  void UniformTrackManager::applyMatchesToFrames(
      const FrameToFrameMatchesWithScore& matches_A_B, VisualFrame* apple_frame,
      VisualFrame* banana_frame) {
    CHECK_NOTNULL(apple_frame);
    CHECK_NOTNULL(banana_frame);

    // Get the track ID channels.
    Eigen::VectorXi& apple_track_ids = *CHECK_NOTNULL(createAndGetTrackIdChannel(apple_frame));
    Eigen::VectorXi& banana_track_ids = *CHECK_NOTNULL(createAndGetTrackIdChannel(banana_frame));
    CHECK(apple_frame->hasKeypointScores());
    CHECK(banana_frame->hasKeypointScores());

    size_t num_apple_track_ids = static_cast<size_t>(apple_track_ids.rows());
    size_t num_banana_track_ids = static_cast<size_t>(banana_track_ids.rows());

    std::unordered_set<int> consumed_apples;
    std::unordered_set<int> consumed_bananas;

    const aslam::Camera::ConstPtr& camera = apple_frame->getCameraGeometry();

    // Prepare buckets.
    std::vector<size_t> buckets;
    buckets.resize(number_of_tracking_buckets_root_ *
                   number_of_tracking_buckets_root_, 0);

    double bucket_width_x = static_cast<double>(camera->imageWidth()) /
        static_cast<double>(number_of_tracking_buckets_root_);
    double bucket_width_y = static_cast<double>(camera->imageHeight()) /
        static_cast<double>(number_of_tracking_buckets_root_);

    std::function<size_t(const Eigen::Vector2d&)> compute_bin_index =
        [&buckets, bucket_width_x, bucket_width_y, this]
         (const Eigen::Vector2d& kp) -> int {
          double bin_x = kp[0] / bucket_width_x;
          double bin_y = kp[1] / bucket_width_y;

          size_t bin_index = static_cast<size_t>(
                              static_cast<int>(std::floor(bin_y)) *
                                number_of_tracking_buckets_root_ +
                              static_cast<int>(std::floor(bin_x)));

          CHECK_LT(bin_index, buckets.size());
          return bin_index;
        };

    std::set<FrameToFrameMatchWithScore, std::greater<MatchWithScore>>
      candidates_for_new_tracks;

    for (const FrameToFrameMatchWithScore& match : matches_A_B) {
      int index_apple = match.getKeypointIndexAppleFrame();
      CHECK_LT(index_apple, static_cast<int>(num_apple_track_ids));

      int index_banana = match.getKeypointIndexBananaFrame();
      CHECK_LT(index_banana, static_cast<int>(num_banana_track_ids));

      addToSetsAndCheckExclusiveness(index_apple,
                                     index_banana,
                                     &consumed_apples,
                                     &consumed_bananas);

      int track_id_apple = apple_track_ids(index_apple);
      int track_id_banana= banana_track_ids(index_banana);

      if ((track_id_apple) < 0 && (track_id_banana < 0)) {
        // Both track ids are < 0. Candidate for a new track.
        FrameToFrameMatchWithScore match_scored_by_keypoint_strenght = match;
        const double apple_keypoint_score =
            apple_frame->getKeypointScores()(index_apple);
        const double banana_keypoint_score =
            banana_frame->getKeypointScores()(index_banana);
        match_scored_by_keypoint_strenght.setScore(
            0.5 * (apple_keypoint_score + banana_keypoint_score));
        candidates_for_new_tracks.emplace(match_scored_by_keypoint_strenght);
      } else {
        // Either one of the track ids is >= 0.
        if (track_id_apple != track_id_banana) {
          // The two track ids are not equal, so we need to copy one over to the
          // other.
          if (track_id_banana >= 0) {
            CHECK_LT(track_id_apple, 0) << "Both the apple and the banana track"
                " id are >= 0 but they are not equal!";

            apple_track_ids(index_apple) = track_id_banana;
          } else {
            CHECK_LT(track_id_banana, 0) << "Both the apple and the banana "
                "track id are >= 0 but they are not equal!";

            banana_track_ids(index_banana) = track_id_apple;
          }
        }
        // Push this match into the buckets.
        const Eigen::Vector2d& keypoint =
            apple_frame->getKeypointMeasurement(index_apple);
        int bin_index = compute_bin_index(keypoint);
        ++buckets[bin_index];
      }
    }
    // Push some number of very strong new track candidates.
    size_t num_very_strong_candidates_pushed = 0u;
    std::set<FrameToFrameMatchWithScore, std::greater<FrameToFrameMatchWithScore>>::
      const_iterator iterator_matches_fo_new_tracks = candidates_for_new_tracks.begin();
    for (; (iterator_matches_fo_new_tracks != candidates_for_new_tracks.end())
        &&  (num_very_strong_candidates_pushed <
            number_of_very_strong_new_tracks_to_force_push_);
              ++iterator_matches_fo_new_tracks,
              ++num_very_strong_candidates_pushed) {
      // The matches are sorted. If we get below the unconditional threshold,
      // we can stop.
      if (iterator_matches_fo_new_tracks->getScore() <
          match_score_very_strong_new_tracks_threshold_) break;

      int index_apple = iterator_matches_fo_new_tracks->getKeypointIndexAppleFrame();
      CHECK_LT(index_apple, static_cast<int>(num_apple_track_ids));

      int index_banana = iterator_matches_fo_new_tracks->getKeypointIndexBananaFrame();
      CHECK_LT(index_banana, static_cast<int>(num_banana_track_ids));

      // Increment the corresponding bucket.
      const Eigen::Vector2d& keypoint =
          apple_frame->getKeypointMeasurement(index_apple);
      int bin_index = compute_bin_index(keypoint);
      ++buckets[bin_index];

      // Write back the applied match.
      int new_track_id = track_id_provider_.getNewId();
      apple_track_ids(index_apple) = new_track_id;
      banana_track_ids(index_banana) = new_track_id;
    }

    // Fill the buckets with the renmaining candidates.
    for (; iterator_matches_fo_new_tracks != candidates_for_new_tracks.end();
        ++iterator_matches_fo_new_tracks) {
      int index_apple = iterator_matches_fo_new_tracks->getKeypointIndexAppleFrame();
      CHECK_LT(index_apple, static_cast<int>(num_apple_track_ids));

      int index_banana = iterator_matches_fo_new_tracks->getKeypointIndexBananaFrame();
      CHECK_LT(index_banana, static_cast<int>(num_banana_track_ids));

      // Get the bucket index and check if there is still space left.
      const Eigen::Vector2d& keypoint =
          apple_frame->getKeypointMeasurement(index_apple);
      int bin_index = compute_bin_index(keypoint);

      if (buckets[bin_index] < bucket_capacity_) {
        ++buckets[bin_index];

        // Write back the applied match.
        int new_track_id = track_id_provider_.getNewId();
        apple_track_ids(index_apple) = new_track_id;
        banana_track_ids(index_banana) = new_track_id;
      }
    }
  }
}
