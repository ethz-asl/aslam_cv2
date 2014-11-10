#ifndef ASLAM_COMMON_UNIQUE_ID_H_
#define ASLAM_COMMON_UNIQUE_ID_H_

#include <sm/hash_id.hpp>

namespace aslam {

#define ASLAM_UNIQUE_ID(IdType) \
class IdType : public sm::HashId { \
 public: \
  IdType() = default; \
}

ASLAM_UNIQUE_ID(FrameId);
ASLAM_UNIQUE_ID(NFramesId);
ASLAM_UNIQUE_ID(CameraId);
ASLAM_UNIQUE_ID(NCameraId);
ASLAM_UNIQUE_ID(LandmarkId);

}  // namespace aslam

SM_DEFINE_HASHID_HASH(aslam::FrameId);
SM_DEFINE_HASHID_HASH(aslam::NFramesId);
SM_DEFINE_HASHID_HASH(aslam::CameraId);
SM_DEFINE_HASHID_HASH(aslam::NCameraId);
SM_DEFINE_HASHID_HASH(aslam::LandmarkId);

#endif  // ASLAM_COMMON_UNIQUE_ID_H_
