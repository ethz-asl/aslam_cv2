#ifndef ASLAM_COMMON_UNIQUE_ID_H_
#define ASLAM_COMMON_UNIQUE_ID_H_
#include <sm/hash_id.hpp>

namespace aslam {
class FrameId : public sm::HashId {
 public:
  FrameId() = default;
};

class MultiFrameId : public sm::HashId {
 public:
  MultiFrameId() = default;
};

class CameraId : public sm::HashId {
 public:
  CameraId() = default;
};

class LandmarkId : public sm::HashId {
 public:
  LandmarkId() = default;
};

}  // namespace aslam

#endif  // ASLAM_COMMON_UNIQUE_ID_H_
