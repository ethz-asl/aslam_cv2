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

class CameraRigId : public sm::HashId {
 public:
  CameraRigId() = default;
};

class LandmarkId : public sm::HashId {
 public:
  LandmarkId() = default;
};

}  // namespace aslam

SM_DEFINE_HASHID_HASH(aslam::FrameId);
SM_DEFINE_HASHID_HASH(aslam::MultiFrameId);
SM_DEFINE_HASHID_HASH(aslam::CameraId);
SM_DEFINE_HASHID_HASH(aslam::CameraRigId);
SM_DEFINE_HASHID_HASH(aslam::LandmarkId);

#endif  // ASLAM_COMMON_UNIQUE_ID_H_
