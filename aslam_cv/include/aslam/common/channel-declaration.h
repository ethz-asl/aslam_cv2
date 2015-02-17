#ifndef ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
#define ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
#include <memory>
#include <string>
#include <unordered_map>

#include <Eigen/Dense>
#include <aslam/common/channel.h>
#include <aslam/common/macros.h>

#define DECLARE_CHANNEL_IMPL(NAME, TYPE)                                   \
namespace aslam {                                                          \
namespace channels {                                                       \
                                                                           \
const std::string NAME##_CHANNEL = #NAME;                                  \
typedef GET_TYPE(TYPE) NAME##_ChannelValueType;                            \
typedef Channel<NAME##_ChannelValueType> NAME##_ChannelType;               \
                                                                           \
NAME##_ChannelValueType& get_##NAME##_Data(const ChannelGroup& channels) { \
  ChannelGroup::const_iterator it = channels.find(NAME##_CHANNEL);         \
  CHECK(it != channels.end()) << "Channelgroup does not "                  \
      "contain channel " << NAME##_CHANNEL;                                \
  std::shared_ptr<NAME##_ChannelType> derived =                            \
     std::dynamic_pointer_cast<NAME##_ChannelType>(it->second);            \
  CHECK(derived) << "Channel cast to derived failed " <<                   \
     "channel: " << NAME##_CHANNEL;                                        \
  return derived->value_;                                                  \
}                                                                          \
                                                                           \
NAME##_ChannelValueType& add_##NAME##_Channel(ChannelGroup* channels) {    \
  CHECK_NOTNULL(channels);                                                 \
  ChannelGroup::iterator it = channels->find(NAME##_CHANNEL);              \
  CHECK(it == channels->end()) << "Channelgroup already "                  \
      "contains channel " << NAME##_CHANNEL;                               \
  std::shared_ptr<NAME##_ChannelType> derived(new NAME##_ChannelType);     \
  (*channels)[NAME##_CHANNEL] = derived;                                   \
  return derived->value_;                                                  \
}                                                                          \
                                                                           \
bool has_##NAME##_Channel(const ChannelGroup& channels) {                  \
  ChannelGroup::const_iterator it = channels.find(NAME##_CHANNEL);         \
  return it != channels.end();                                             \
}                                                                          \
}                                                                          \
}                                                                          \

// Wrap types that contain commas inside braces.
#define DECLARE_CHANNEL(x, ...) DECLARE_CHANNEL_IMPL(x, (__VA_ARGS__))

namespace aslam {
namespace channels {
template<typename CHANNEL_DATA_TYPE>
CHANNEL_DATA_TYPE& getChannelData(const std::string& channelName,
                                  const ChannelGroup& channels) {
  ChannelGroup::const_iterator it = channels.find(channelName);
  CHECK(it != channels.end()) << "Channelgroup does not "
      "contain channel " << channelName;
  typedef Channel<CHANNEL_DATA_TYPE> DerivedChannel;
  std::shared_ptr<DerivedChannel> derived =
      std::dynamic_pointer_cast < DerivedChannel > (it->second);
  CHECK(derived) << "Channel cast to derived failed " <<
                    "channel: " << channelName;
  return derived->value_;
}

inline bool hasChannel(const std::string& channelName,
                       const ChannelGroup& channels) {
  ChannelGroup::const_iterator it = channels.find(channelName);
  return it != channels.end();
}

template<typename CHANNEL_DATA_TYPE>
CHANNEL_DATA_TYPE& addChannel(const std::string& channelName,
                              ChannelGroup* channels) {
  CHECK_NOTNULL(channels);
  ChannelGroup::iterator it = channels->find(channelName);
  CHECK(it == channels->end()) << "Channelgroup already "
      "contains channel " << channelName;
  typedef Channel<CHANNEL_DATA_TYPE> DerivedChannel;
  std::shared_ptr < DerivedChannel > derived(new DerivedChannel);
  (*channels)[channelName] = derived;
  return derived->value_;
}
}  // namespace channels
}  // namespace aslam

#endif  // ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
