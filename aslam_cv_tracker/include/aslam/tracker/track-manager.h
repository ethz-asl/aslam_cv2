#ifndef ASLAM_TRACK_MANAGER_H_
#define ASLAM_TRACK_MANAGER_H_

#include <mutex>
#include <unordered_set>

#include <aslam/matcher/match.h>
#include <glog/logging.h>

namespace aslam {
  struct MatchWithScore;
  class VisualNFrame;

  template<typename IdType>
  class ThreadSafeIdProvider {
   public:
    ThreadSafeIdProvider(IdType initial_id) : initial_id_(initial_id) {
      reset();
    }

    IdType getNewId() {
      std::lock_guard<std::mutex> lock(mutex_);
      return id_++;
    }

    void reset() {
      std::lock_guard<std::mutex> lock(mutex_);
      id_ = initial_id_;
    }

   private:
    std::mutex mutex_;
    IdType id_;
    IdType initial_id_;
  };

  /// \brief The Track manager assigns track ids to the given matches with different strategies.
  class TrackManager {
   public:
    TrackManager() {}
    virtual ~TrackManager() {};

    /// \brief Writes track ids for a list of matches into two given frames.
    ///
    /// @param[in]  matches_A_B   List of matches between the Apple and Banana frame.
    /// @param[in]  apple_frame   Pointer to the apple frame.
    /// @param[in]  banana_frame  Pointer to the banana frame.
    virtual void applyMatchesToFrames(
        const FrameToFrameMatchesWithScore& matches_A_B,
        VisualFrame* apple_frame, VisualFrame* banana_frame) = 0;

    /// \brief Returns a pointer to the track id channel. If no track id channel is present for the
    ///        given frame, a new track id channel will be created with num_keypoints many track
    ///        ids, all set to to -1.
    ///
    /// @param[in]  frame   Pointer to the visual frame.
    /// @return             Pointer to the track id channel.
    static Eigen::VectorXi* createAndGetTrackIdChannel(VisualFrame* frame);

    static void resetIdProvider() {
      track_id_provider_.reset();
    }

   protected:
    static ThreadSafeIdProvider<size_t> track_id_provider_;
  };


  /// \brief Track manager simply writing track ids into the given frames for
  ///        the given matches.
  class SimpleTrackManager : public TrackManager {
   public:
    SimpleTrackManager() = default;
    virtual ~SimpleTrackManager() {};

    /// \brief Writes track ids into the given frames for the given matches.
    ///        If for a match, both track ids are < 0, a new track id is
    ///        generated and applied.
    ///        If any of the two track ids for a match is >= 0 the other one is
    ///        either expected to be identical (in which case no change is
    ///        applied) or < 0, in which case the valid id (>=0) is copied over.
    ///        Matches are expected to be exclusive.
    virtual void applyMatchesToFrames(
        const FrameToFrameMatchesWithScore& matches,
        VisualFrame* apple_frame, VisualFrame* banana_frame);
  };

  /// \brief Track manager using buckets to uniformly distribute weak new
  ///        tracks. The image space of the apple frame is divided into
  ///        num_buckets_root^2 buckets.
  ///
  ///        Note: max_number_of_weak_new_tracks is only an upper bound for the
  ///        max number of non-strong new tracks that can be born.
  ///        It determines the bucket capacity as follows:
  ///   bucket_capacity := floor(max_number_of_weak_new_tracks / num_buckets)
  ///        It does not influence the number of strong new tracks that are
  ///        forced pushed. I.e. there may be more strong new tracks being born
  ///        in a bucket than bucket_capacity. In this case, no further weak
  ///        new tracks will be generated in this particular bucket but it does
  ///        not reduce the number of weak new tracks accepted in other (not yet
  ///        full) buckets.
  class UniformTrackManager : public TrackManager {
   public:
    UniformTrackManager(size_t num_buckets_root,
                        size_t max_number_of_weak_new_tracks,
                        size_t num_strong_new_tracks_to_force_push,
                        double match_score_very_strong_new_tracks_threshold) :
      number_of_tracking_buckets_root_(num_buckets_root),
      bucket_capacity_(std::floor(
          static_cast<double>(max_number_of_weak_new_tracks) /
          static_cast<double>(num_buckets_root * num_buckets_root))),
      number_of_very_strong_new_tracks_to_force_push_(
          num_strong_new_tracks_to_force_push),
      match_score_very_strong_new_tracks_threshold_(
          match_score_very_strong_new_tracks_threshold) {}

    virtual ~UniformTrackManager() {};

    /// \brief Writes track ids into the given frames for the given matches.
    ///        In a first iteration, all track ids from preexisting tracks
    ///        get assigned (i.e. matches, for which either one of the track ids
    ///        is valid (>=0)).
    ///        In a second iteration, the
    ///        number_of_very_strong_new_tracks_to_push_ strongest new track
    ///        matches are applied. The buckets get filled accordingly and may
    ///        even overflow if the number of strong matches in a bucket
    ///        exceeds the bucket capacity.
    ///        In a third iteration, all buckets with remaining capacity are
    ///        filled with the next best matches until either all buckets are
    ///        full or all matches have been accepted.
    ///        The matches are expected to be exclusive.
    virtual void applyMatchesToFrames(
        const FrameToFrameMatchesWithScore& matches_A_B,
        VisualFrame* apple_frame, VisualFrame* banana_frame);
   private:
    /// \brief Square root of the number of tracking buckets. The image space
    ///        gets devided into number_of_tracking_buckets_root_^2 buckets.
    size_t number_of_tracking_buckets_root_;
    /// \brief Max number of weak new tracks per bucket.
    size_t bucket_capacity_;
    /// \brief Number of very strong new tracks to push.
    ///        Of all new born tracks, the
    ///        number_of_very_strong_new_tracks_to_push_ strongest (best score)
    ///        matches are applied, irregardless of the bucket levels.
    size_t number_of_very_strong_new_tracks_to_force_push_;
    /// \brief Match score threshold for the very strong new tracks.
    double match_score_very_strong_new_tracks_threshold_;
  };

  inline void addToSetsAndCheckExclusiveness(
      int index_apple, int index_banana, std::unordered_set<int>* consumed_apples,
      std::unordered_set<int>* consumed_bananas) {
    CHECK_NOTNULL(consumed_apples);
    CHECK_NOTNULL(consumed_bananas);
    std::pair<std::unordered_set<int>::iterator, bool> ret_apple =
        consumed_apples->insert(index_apple);
    CHECK(ret_apple.second) << "The given matches don't seem to be exclusive."
        " Trying to assign apple " << index_apple << " more than once!";
    std::pair<std::unordered_set<int>::iterator, bool> ret_banana =
        consumed_bananas->insert(index_banana);
    CHECK(ret_banana.second) << "The given matches don't seem to be "
        "exclusive. Trying to assign banana " << index_banana << " more than "
            "once!";

  }

}  // namespace aslam

#endif  // ASLAM_TRACK_MANAGER_H_
