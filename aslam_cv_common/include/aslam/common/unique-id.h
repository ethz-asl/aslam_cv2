#ifndef ASLAM_COMMON_UNIQUE_ID_H_
#define ASLAM_COMMON_UNIQUE_ID_H_

#include <unordered_set>
#include <vector>

#include <aslam/common/hash-id.h>

namespace aslam {
#define ASLAM_UNIQUE_ID(IdType)                   \
  class IdType : public HashId {                  \
   public:                                        \
    inline static IdType Random() {               \
      IdType generated;                           \
      generated.randomize();                      \
      return generated;                           \
    }                                             \
    IdType() = default;                           \
  };                                              \
  typedef std::vector<IdType> IdType##List;       \
  typedef std::unordered_set<IdType> IdType##Set

ASLAM_UNIQUE_ID(FrameId);
ASLAM_UNIQUE_ID(NFramesId);
ASLAM_UNIQUE_ID(CameraId);
ASLAM_UNIQUE_ID(NCameraId);

}  // namespace aslam

ASLAM_DEFINE_HASHID_HASH(aslam::FrameId);
ASLAM_DEFINE_HASHID_HASH(aslam::NFramesId);
ASLAM_DEFINE_HASHID_HASH(aslam::CameraId);
ASLAM_DEFINE_HASHID_HASH(aslam::NCameraId);

#endif  // ASLAM_COMMON_UNIQUE_ID_H_
