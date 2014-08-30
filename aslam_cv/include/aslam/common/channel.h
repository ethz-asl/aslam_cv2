#ifndef ASLAM_CV_COMMON_CHANNEL_H_
#define ASLAM_CV_COMMON_CHANNEL_H_

/// \addtogroup Frames
/// @{
/// \defgroup Channels
/// @{
///
/// Channels are key/value pairs that can be added to a frame. We have support
/// for several common channel types like Eigen::Matrices.
///
/// To implement a new channel type simply...
/// TODO(slynen) please describe what to do to implement a new channel type
///
/// @}
/// @}

#include <string>
#include <unordered_map>

#include <aslam/common/macros.h>
#include <aslam/common/channel-serialization.h>

namespace aslam {
namespace channels {
class ChannelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(ChannelBase);
  ChannelBase() {}
  virtual ~ChannelBase() {};
  virtual bool serializeToString(std::string* string) const = 0;
  virtual bool deSerializeFromString(const std::string& string) = 0;
  virtual bool serializeToBuffer(char** buffer, size_t* size) const = 0;
  virtual bool deSerializeFromBuffer(const char* const buffer, size_t size) = 0;
  virtual std::string name() const = 0;
};

template<typename TYPE>
class Channel : public ChannelBase{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef TYPE Type;
  Channel() {}
  virtual ~Channel() {}
  virtual std::string name() const { return "unnamed"; }
  bool operator==(const Channel<TYPE>& other) {
    return value_ == other.value_;
  }
  bool serializeToString(std::string* string) const {
    return aslam::internal::serializeToString(value_, string);
  }
  bool serializeToBuffer(char** buffer, size_t* size) const {
    return aslam::internal::serializeToBuffer(value_, buffer, size);
  }
  bool deSerializeFromString(const std::string& string) {
    return aslam::internal::deSerializeFromString(string, &value_);
  }
  bool deSerializeFromBuffer(const char* const buffer, size_t size) {
    return aslam::internal::deSerializeFromBuffer(buffer, size, &value_);
  }
  TYPE value_;
};

typedef std::unordered_map<std::string, std::shared_ptr<ChannelBase> > ChannelGroup;
}  // namespace channels
}  // namespace aslam
#endif  // ASLAM_CV_COMMON_CHANNEL_H_
