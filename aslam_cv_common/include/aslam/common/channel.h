#ifndef ASLAM_CV_COMMON_CHANNEL_H_
#define ASLAM_CV_COMMON_CHANNEL_H_
#include <string>

#include <aslam/common/macros.h>
#include <aslam/common/channel-serialization.h>

namespace aslam {
class ChannelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(ChannelBase);
  ChannelBase() {}
  virtual ~ChannelBase() {};
  virtual bool serializeToString(std::string* string) const = 0;
  virtual bool deSerializeFromString(const std::string& string) = 0;
  virtual bool serializeToString(char** buffer, size_t* size) const = 0;
  virtual bool deSerializeFromString(const char* const buffer, size_t size) = 0;
  virtual std::string name() const = 0;
};

template<typename TYPE>
class Channel : public ChannelBase{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef TYPE Type;
  Channel() {}
  virtual ~Channel() {}
  virtual std::string name() const { return "unnamed"; }
  bool operator==(const Channel<TYPE>& other) {
    return value_ == other.value_;
  }
  bool serializeToString(std::string* string) const {
    aslam::internal::serializeToString(value_, string);
  }
  bool serializeToString(char** buffer, size_t* size) const {
    aslam::internal::serializeToString(value_, buffer, size);
  }
  bool deSerializeFromString(const std::string& string) {
    aslam::internal::deSerializeFromString(string, &value_);
  }
  bool deSerializeFromString(const char* const buffer, size_t size) {
    aslam::internal::deSerializeFromString(buffer, size, &value_);
  }
  TYPE value_;
};
}  // namespace aslam
#endif  // ASLAM_CV_COMMON_CHANNEL_H_
