#ifndef ASLAM_TRACK_MANAGER_H_
#define ASLAM_TRACK_MANAGER_H_

namespace aslam {
  class VisualFrame;
  class Match;

  /// \brief The Track manager assigns track IDs and allows applying matches
  ///        with different strategies.
  class TrackManager {
   public:
    TrackManager() : track_id_provider_(0) {}
    explicit TrackManager(size_t track_id);
    virtual ~TrackManager() {};

    /// \brief Applies a list of matches to two given frames.
    ///        The track ids of the given frames will be altered accordingly.
    ///
    /// @param[in]  apple_frame   Shared pointer to the apple frame.
    /// @param[in]  banna_frame   Shared pointer to the banana frame.
    /// @param[in]  matches       List of matches between the two given frames.
    virtual void applyMatchesToFrames(
                              const std::shared_ptr<VisualFrame>& apple_frame,
                              const std::shared_ptr<VisualFrame>& banana_frame,
                              const std::vector<Match>& matches) = 0;
   protected:
    size_t track_id_provider_;
  };


  /// \brief Track manager simply writing track ids into the given frames for
  ///        the given matches.
  class SimpleTrackManager : public TrackManager {
   public:
    SimpleTrackManager() = default;
    explicit SimpleTrackManager(size_t start_track_id) :
        TrackManager(start_track_id) {};
    virtual ~SimpleTrackManager() {};

    /// \brief Writes track ids into the given frames for the given matches.
    ///        If for a match, both track ids are < 0, a new track id is
    ///        generated and applied.
    ///        If any of the two track ids for a match is >= 0 the other one is
    ///        either expected to be identical (in which case no change is
    ///        applied) or < 0, in which case the valid id (>=0) is copied over.
    ///        Matches are expected to be exclusive.
    ///
    /// @param[in]  apple_frame   Shared pointer to the apple frame.
    /// @param[in]  banna_frame   Shared pointer to the banana frame.
    /// @param[in]  matches       List of matches between the two given frames.
    virtual void applyMatchesToFrames(
                            const std::shared_ptr<VisualFrame>& apple_frame,
                            const std::shared_ptr<VisualFrame>& banana_frame,
                            const std::vector<Match>& matches);
  };

  /// \brief Track manager using buckets to uniformly distribute weak new
  ///        tracks. The image space of the apple frame is divided into
  ///        num_buckets_root^2 buckets.
  class UniformTrackManager : public TrackManager {
   public:
    UniformTrackManager(size_t num_buckets_root,
                        size_t bucket_capacity,
                        size_t num_strong_new_tracks_to_push,
                        double match_score_very_strong_new_tracks_threshold) :
      number_of_tracking_buckets_root_(num_buckets_root),
      bucket_capacity_(bucket_capacity),
      number_of_very_strong_new_tracks_to_push_(
          num_strong_new_tracks_to_push),
      match_score_very_strong_new_tracks_threshold_(
          match_score_very_strong_new_tracks_threshold) {}
    explicit UniformTrackManager(
                           size_t start_track_id,
                           size_t num_buckets,
                           size_t bucket_capacity,
                           size_t num_strong_new_tracks_to_push,
                           double match_score_very_strong_new_tracks_threshold)
    : number_of_tracking_buckets_root_(num_buckets),
      bucket_capacity_(bucket_capacity),
      number_of_very_strong_new_tracks_to_push_(num_strong_new_tracks_to_push),
      match_score_very_strong_new_tracks_threshold_(
          match_score_very_strong_new_tracks_threshold),
      TrackManager(start_track_id) {};
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
    ///        In a third iteration, all buckets with remaining capactiy are
    ///        filled with the next best matches until either all buckets are
    ///        full or all matches have bee napplied.
    ///        The matches are expected to be exclusive.
    ///
    /// @param[in]  apple_frame   Shared pointer to the apple frame.
    /// @param[in]  banna_frame   Shared pointer to the banana frame.
    /// @param[in]  matches       List of matches between the two given frames.
    virtual void applyMatchesToFrames(
                            const std::shared_ptr<VisualFrame>& apple_frame,
                            const std::shared_ptr<VisualFrame>& banana_frame,
                            const std::vector<Match>& matches);

    /////////////
    /// Setters for the parameters
    /////////////
    void setNumberOfStrongNewTracksToPush(
        size_t number_of_strong_new_tracks_to_push);
    void setNumberOfTrackingBucketsRoot(
        size_t number_of_tracking_buckets);
    void setBucketCapacity(size_t bucket_capacity);
    void setKeypointScoreThresholdUnconditional(
        double keypoint_score_threshold_unconditional);

   private:
    /// \brief Square root of the number of tracking buckets. The image space
    ///        gets devided into number_of_tracking_buckets_root_^2 buckets.
    size_t number_of_tracking_buckets_root_;
    /// \brief Max number of matches per bucket.
    size_t bucket_capacity_;
    /// \brief Number of very strong new tracks to push.
    ///        Of all new born tracks, the
    ///        number_of_very_strong_new_tracks_to_push_ strongest (best score)
    ///        matches are applied, irregardless of the bucket levels.
    size_t number_of_very_strong_new_tracks_to_push_;
    /// \brief Match score threshold for the very strong new tracks.
    double match_score_very_strong_new_tracks_threshold_;
  };
}

#endif  // ASLAM_TRACK_MANAGER_H_
