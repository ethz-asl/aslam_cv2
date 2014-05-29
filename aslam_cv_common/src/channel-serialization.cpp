#include <glog/logging.h>
#include <aslam/common/channel-serialization.h>

namespace aslam {
namespace internal {
size_t HeaderInformation::size() const {
  return sizeof(rows) + sizeof(cols) + sizeof(type);
}

bool HeaderInformation::serializeToBuffer(char* buffer, size_t offset) const {
  CHECK_NOTNULL(buffer);
  buffer += offset;
  memcpy(buffer, &rows, sizeof(rows));
  buffer += sizeof(rows);
  memcpy(buffer, &cols, sizeof(cols));
  buffer += sizeof(cols);
  memcpy(buffer, &type, sizeof(type));
  buffer += sizeof(type);
  return true;
}

bool HeaderInformation::deSerializeFromBuffer(const char* const buffer_in, size_t offset) {
  CHECK_NOTNULL(buffer_in);
  const char* buffer = buffer_in;
  buffer += offset;
  memcpy(&rows, buffer, sizeof(rows));
  buffer += sizeof(rows);
  memcpy(&cols, buffer, sizeof(cols));
  buffer += sizeof(cols);
  memcpy(&type, buffer, sizeof(type));
  buffer += sizeof(type);
  return true;
}
}  // namespace internal
}  // namespace aslam
