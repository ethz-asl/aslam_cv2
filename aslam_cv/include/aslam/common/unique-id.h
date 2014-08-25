#ifndef ASLAM_COMMON_UNIQUE_ID_H_
#define ASLAM_COMMON_UNIQUE_ID_H_
#include <sm/hash_id.hpp>

namespace aslam {
class FrameId : public sm::HashId {
 public:
  FrameId() = default;
};

class NFramesId : public sm::HashId {
 public:
  NFramesId() = default;
};

class CameraId : public sm::HashId {
 public:
  CameraId() = default;
};

class NCamerasId : public sm::HashId {
 public:
  NCamerasId() = default;
};

class LandmarkId : public sm::HashId {
 public:
  LandmarkId() = default;
};

}  // namespace aslam

SM_DEFINE_HASHID_HASH(aslam::FrameId);
SM_DEFINE_HASHID_HASH(aslam::NFramesId);
SM_DEFINE_HASHID_HASH(aslam::CameraId);
SM_DEFINE_HASHID_HASH(aslam::NCamerasId);
SM_DEFINE_HASHID_HASH(aslam::LandmarkId);

#endif  // ASLAM_COMMON_UNIQUE_ID_H_
