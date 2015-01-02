#include <set>

#include <glog/logging.h>

#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/match.h>

#include "aslam/tracker/track-manager.h"

namespace aslam {
  TrackManager::TrackManager(size_t start_track_id) {
    track_id_provider_ = start_track_id;
  }

  void SimpleTrackManager::applyMatchesToFrames(const Matches& matches,
                                                VisualFrame* apple_frame,
                                                VisualFrame* banana_frame) {
    CHECK_NOTNULL(apple_frame);
    CHECK_NOTNULL(banana_frame);

    Eigen::VectorXi& apple_track_ids =
        *CHECK_NOTNULL(apple_frame->getTrackIdsMutable());
    size_t num_apple_track_ids = apple_track_ids.rows();

    Eigen::VectorXi& banana_track_ids =
        *CHECK_NOTNULL(banana_frame->getTrackIdsMutable());
    size_t num_banana_track_ids = banana_track_ids.rows();

    std::unordered_set<int> consumed_apples;
    std::unordered_set<int> consumed_bananas;

    for (const Match& match : matches) {
      int index_apple = match.getIndexApple();
      CHECK_LT(index_apple, num_apple_track_ids);

      int index_banana = match.getIndexBanana();
      CHECK_LT(index_banana, num_banana_track_ids);

      addToSetsAndCheckExclusiveness(index_apple,
                                     index_banana,
                                     &consumed_apples,
                                     &consumed_bananas);

      int track_id_apple = apple_track_ids(index_apple);
      int track_id_banana= banana_track_ids(index_banana);

      if ((track_id_apple) < 0 && (track_id_banana < 0)) {
        // Both track ids are < 0. Start a new track.
        int new_track_id = track_id_provider_++;
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

  void UniformTrackManager::applyMatchesToFrames(const Matches& matches,
                                                 VisualFrame* apple_frame,
                                                 VisualFrame* banana_frame) {
    CHECK_NOTNULL(apple_frame);
    CHECK_NOTNULL(banana_frame);

    Eigen::VectorXi& apple_track_ids =
        *CHECK_NOTNULL(apple_frame->getTrackIdsMutable());
    size_t num_apple_track_ids = apple_track_ids.rows();

    Eigen::VectorXi& banana_track_ids =
            *CHECK_NOTNULL(banana_frame->getTrackIdsMutable());
    size_t num_banana_track_ids = banana_track_ids.rows();

    std::unordered_set<int> consumed_apples;
    std::unordered_set<int> consumed_bananas;

    const aslam::Camera::ConstPtr& camera = apple_frame->getCameraGeometry();

    // Prepare buckets.
    std::vector<int> buckets;
    buckets.resize(number_of_tracking_buckets_root_ *
                   number_of_tracking_buckets_root_, 0);

    double bucket_width_x = static_cast<double>(camera->imageWidth()) /
        static_cast<double>(number_of_tracking_buckets_root_);
    double bucket_width_y = static_cast<double>(camera->imageHeight()) /
        static_cast<double>(number_of_tracking_buckets_root_);

    std::function<size_t(const Eigen::Vector2d&)> compute_bin_index =
        [buckets, bucket_width_x, bucket_width_y, this]
         (const Eigen::Vector2d& kp) -> int {
          double bin_x = kp[0] / bucket_width_x;
          double bin_y = kp[1] / bucket_width_y;

          size_t bin_index = static_cast<size_t>(
                              static_cast<int>(std::floor(bin_y)) *
                                number_of_tracking_buckets_root_ +
                              static_cast<int>(std::floor(bin_x)));

          CHECK_LT(bin_index, static_cast<int>(buckets.size()));
          return bin_index;
        };

    std::set<Match, std::greater<Match> > candidates_for_new_tracks;

    for (const Match& match : matches) {
      int index_apple = match.getIndexApple();
      CHECK_LT(index_apple, num_apple_track_ids);

      int index_banana = match.getIndexBanana();
      CHECK_LT(index_banana, num_banana_track_ids);

      addToSetsAndCheckExclusiveness(index_apple,
                                     index_banana,
                                     &consumed_apples,
                                     &consumed_bananas);

      int track_id_apple = apple_track_ids(index_apple);
      int track_id_banana= banana_track_ids(index_banana);

      if ((track_id_apple) < 0 && (track_id_banana < 0)) {
        // Both track ids are < 0. Candidate for a new track.
        candidates_for_new_tracks.insert(match);
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
      }
    }
    // Push some number of very strong new track candidates.
    size_t num_very_strong_candidates_pushed = 0u;
    auto iterator_matches_fo_new_tracks = candidates_for_new_tracks.begin();
    for (; (iterator_matches_fo_new_tracks != candidates_for_new_tracks.end())
        &&  (num_very_strong_candidates_pushed <
            number_of_very_strong_new_tracks_to_force_push_);
              ++iterator_matches_fo_new_tracks,
              ++num_very_strong_candidates_pushed) {
      // The matches are sorted. If we get below the unconditional threshold,
      // we can stop.
      if (iterator_matches_fo_new_tracks->score <
          match_score_very_strong_new_tracks_threshold_) break;

      int index_apple = iterator_matches_fo_new_tracks->getIndexApple();
      CHECK_LT(index_apple, num_apple_track_ids);

      int index_banana = iterator_matches_fo_new_tracks->getIndexBanana();
      CHECK_LT(index_banana, num_banana_track_ids);

      // Increment the corresponding bucket.
      const Eigen::Vector2d& keypoint =
          apple_frame->getKeypointMeasurement(index_apple);
      int bin_index = compute_bin_index(keypoint);
      ++buckets[bin_index];

      // Write back the applied match.
      int new_track_id = track_id_provider_++;
      apple_track_ids(index_apple) = new_track_id;
      banana_track_ids(index_banana) = new_track_id;
    }

    // Fill the buckets with the renmaining candidates.
    for (; iterator_matches_fo_new_tracks != candidates_for_new_tracks.end();
        ++iterator_matches_fo_new_tracks) {
      int index_apple = iterator_matches_fo_new_tracks->getIndexApple();
      CHECK_LT(index_apple, num_apple_track_ids);

      int index_banana = iterator_matches_fo_new_tracks->getIndexBanana();
      CHECK_LT(index_banana, num_banana_track_ids);

      // Get the bucket index and check if there is still space left.
      const Eigen::Vector2d& keypoint =
          apple_frame->getKeypointMeasurement(index_apple);
      int bin_index = compute_bin_index(keypoint);

      if (buckets[bin_index] < bucket_capacity_) {
        ++buckets[bin_index];

        // Write back the applied match.
        int new_track_id = track_id_provider_++;
        apple_track_ids(index_apple) = new_track_id;
        banana_track_ids(index_banana) = new_track_id;
      }
    }
  }
}
