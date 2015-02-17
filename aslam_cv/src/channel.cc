#include <string>
#include <unordered_map>

#include <aslam/common/channel.h>
#include <aslam/common/meta.h>

namespace aslam {
namespace channels {

template<>
bool Channel<cv::Mat>::operator==(const Channel<cv::Mat>& other) {
  return cv::countNonZero(value_ != other.value_) == 0;
}

ChannelGroup cloneChannelGroup(const ChannelGroup& channels) {
  ChannelGroup cloned_channels;
  for (const ChannelGroup::value_type& channel : channels) {
    cloned_channels.emplace(channel.first,
                            std::shared_ptr<ChannelBase>(channel.second->clone()));
  }
  return cloned_channels;
}

bool isChannelGroupEqual(const ChannelGroup& left_channels, const ChannelGroup& right_channels) {
  if (left_channels.size() != right_channels.size()) {
    return false;
  }
  for (const ChannelGroup::value_type& left_channel_pair : left_channels) {
    ChannelGroup::const_iterator it_right = right_channels.find(left_channel_pair.first);
    if (it_right == right_channels.end()) {
      return false;
    }
    if (!it_right->second->compare(*left_channel_pair.second)) {
      return false;
    }
  }
  return true;
}

}  // namespace channels
}  // namespace aslam
