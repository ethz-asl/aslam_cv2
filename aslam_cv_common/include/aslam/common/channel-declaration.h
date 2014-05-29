#ifndef ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
#define ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
#include <memory>
#include <string>
#include <unordered_map>

#include <Eigen/Dense>
#include <aslam/common/channel.h>
#include <aslam/common/macros.h>

namespace aslam {
namespace channels {
typedef std::unordered_map<std::string, std::shared_ptr<aslam::ChannelBase> > ChannelGroup;
}  // namespace channels
}  // namespace aslam

#define DECLARE_CHANNEL_IMPL(NAME, TYPE)                     \
namespace aslam {                                            \
namespace channels {                                         \
struct NAME : aslam::Channel<GET_TYPE(TYPE)> {               \
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;                           \
  typedef typename GET_TYPE(TYPE) Type;                      \
  virtual std::string name() const { return #NAME; }         \
};                                                           \
                                                             \
const std::string NAME##_CHANNEL = #NAME;                    \
typedef GET_TYPE(TYPE) NAME##_ChannelType;                   \
                                                             \
NAME##_ChannelType& get_##NAME##_Data(const ChannelGroup& channels) { \
  ChannelGroup::const_iterator it = channels.find(NAME##_CHANNEL); \
  CHECK(it != channels.end()) << "Channelgroup does not "    \
      "contain channel " << NAME##_CHANNEL;                  \
  std::shared_ptr<NAME> derived =                            \
     std::dynamic_pointer_cast<NAME>(it->second);            \
  CHECK(derived) << "Channel cast to derived failed " <<     \
     "channel: " << NAME##_CHANNEL;                          \
  return derived->value_;                                    \
}                                                            \
                                                             \
NAME##_ChannelType& add_##NAME##_Channel(ChannelGroup* channels) { \
  CHECK_NOTNULL(channels);                                   \
  ChannelGroup::iterator it = channels->find(NAME##_CHANNEL);\
  CHECK(it == channels->end()) << "Channelgroup already "    \
      "contains channel " << NAME##_CHANNEL;                 \
  std::shared_ptr<NAME> derived(new NAME);                   \
  (*channels)[NAME##_CHANNEL] = derived;                     \
  return derived->value_;                                    \
}                                                            \
                                                             \
bool has_##NAME##_Channel(const ChannelGroup& channels) {    \
  ChannelGroup::const_iterator it = channels.find(NAME##_CHANNEL); \
  return it != channels.end();                               \
}                                                            \
}                                                            \
}                                                            \

// Wrap types that contain commas inside braces.
#define DECLARE_CHANNEL(x, ...) DECLARE_CHANNEL_IMPL(x, (__VA_ARGS__))

#endif  // ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
