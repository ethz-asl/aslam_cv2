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

#include <aslam/common/channel-serialization.h>
#include <aslam/common/crtp-clone.h>
#include <aslam/common/macros.h>
#include <aslam/common/meta.h>

namespace aslam {
namespace channels {

namespace internal {
template<typename ValueType, typename ValueTypeClonable>
struct ChannelValueClonerImpl {};
template<typename ValueType>
struct ChannelValueClonerImpl<ValueType, std::false_type> {
  static ValueType clone(const ValueType& value) { return value; }
};
template<typename ValueType>
struct ChannelValueClonerImpl<ValueType, std::true_type> {
  static ValueType clone(const ValueType& value) { return value.clone(); }
};
template<typename ValueType>
struct ChannelValueCloner :
    public ChannelValueClonerImpl<ValueType, typename is_cloneable<ValueType>::type> {};
}

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
  virtual ChannelBase* clone() const = 0;
  virtual bool compare(const ChannelBase& right) = 0;
};

template<typename TYPE>
class Channel : public aslam::Cloneable<ChannelBase, Channel<TYPE>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ASLAM_POINTER_TYPEDEFS(Channel);

  typedef TYPE Type;
  Channel() {}
  virtual ~Channel() {}
  virtual std::string name() const { return "unnamed"; }
  bool operator==(const Channel& other);

  Channel(const Channel<TYPE>& other) {
    value_ = internal::ChannelValueCloner<TYPE>::clone(other.value_);
  }
  void operator=(const Channel<TYPE>&) = delete;
  virtual bool compare(const ChannelBase& other) {
    return *this == *CHECK_NOTNULL(dynamic_cast<const Channel<TYPE>*>(&other));
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

template<> bool Channel<cv::Mat>::operator==(const Channel<cv::Mat>& other);
template<typename TYPE>
bool Channel<TYPE>::operator==(const Channel<TYPE>& other) {
  static_assert(is_not_pointer<TYPE>::value, "Can't be pointer!");
  return value_ == other.value_;
}

typedef std::unordered_map<std::string, std::shared_ptr<ChannelBase> > ChannelGroup;

ChannelGroup cloneChannelGroup(const ChannelGroup& channels);
bool isChannelGroupEqual(const ChannelGroup& left, const ChannelGroup& right);

}  // namespace channels
}  // namespace aslam
#endif  // ASLAM_CV_COMMON_CHANNEL_H_
