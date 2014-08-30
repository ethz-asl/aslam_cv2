#include <glog/logging.h>
#include <aslam/common/channel-serialization.h>

namespace aslam {
namespace internal {
size_t HeaderInformation::size() const {
  return sizeof(rows) + sizeof(cols) + sizeof(type) + sizeof(channels);
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
  memcpy(buffer, &channels, sizeof(channels));
  buffer += sizeof(channels);
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
  memcpy(&channels, buffer, sizeof(channels));
  buffer += sizeof(channels);
  return true;
}


bool serializeToString(const cv::Mat& image,
                       std::string* string) {

  CHECK(image.isContinuous()) << "This method only works if the image is stored "
      "in contiguous memory.";
  CHECK_EQ(image.total(), static_cast<size_t>(image.rows * image.cols * image.channels())) <<
      "Unexpected number of pixels in the image.";
  CHECK_EQ(image.dims, 2) << "This method only works for 2D arrays";
  const char* const matrixData = reinterpret_cast<const char*>(image.data);
  bool success = false;
  // http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-depth
  switch(image.type()) {
    case cv::DataType<uint8_t>::type:
    success = serializeToString<uint8_t>(matrixData, image.rows,
                                         image.cols, image.channels(),
                                         string);
    case cv::DataType<int8_t>::type:
    success = serializeToString<int8_t>(matrixData, image.rows,
                                        image.cols, image.channels(),
                                        string);
    break;
    case cv::DataType<uint16_t>::type:
    success = serializeToString<uint16_t>(matrixData, image.rows,
                                          image.cols, image.channels(),
                                          string);
    break;
    case cv::DataType<int16_t>::type:
    success = serializeToString<int16_t>(matrixData, image.rows,
                                         image.cols, image.channels(),
                                         string);
    break;
    case cv::DataType<int>::type:
    success = serializeToString<int>(matrixData, image.rows,
                                     image.cols, image.channels(),
                                     string);
    break;
    case cv::DataType<double>::type:
    success = serializeToString<double>(matrixData, image.rows,
                                        image.cols, image.channels(),
                                        string);
    break;
    case cv::DataType<float>::type:
    success = serializeToString<float>(matrixData, image.rows,
                                       image.cols, image.channels(),
                                       string);
    break;
    default:
      LOG(ERROR) << "cv::Mat type " << image.type() << " is not supported for "
      << "serialization.";
      success = false;
      break;
  }
  return success;
}

bool deSerializeFromString(const std::string& string, cv::Mat* image) {
  CHECK_NOTNULL(image);
  return deSerializeFromBuffer(string.data(), string.size(), image);
}

template<typename SCALAR>
bool deSerializeTypedFromBuffer(const char* const buffer, size_t size,
                                const HeaderInformation& header, cv::Mat* image) {
  size_t matrix_size = sizeof(SCALAR) * header.rows * header.cols * header.channels;
  size_t total_size = matrix_size + header.size();
  CHECK_EQ(size, total_size);
  int type = CV_MAKETYPE(cv::DataType<SCALAR>::depth, header.channels);
  // http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-create
  // Create should only allocate if necessary.
  image->create(header.rows, header.cols, type);
  memcpy(image->data, buffer + header.size(), matrix_size);
  return true;
}

bool deSerializeFromBuffer(const char* const buffer, size_t size, cv::Mat* image) {
  CHECK_NOTNULL(image);
  HeaderInformation header;
  CHECK_GE(size, header.size());
  bool success = header.deSerializeFromBuffer(buffer, 0);
  if (!success) {
    LOG(ERROR) << "Failed to deserialize header from string: " <<
        std::string(buffer, size);
    return false;
  }

  // http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-depth
  switch(header.type) {
    case cv::DataType<uint8_t>::type:
    success = deSerializeTypedFromBuffer<uint8_t>(buffer, size, header, image);
    break;
    case cv::DataType<int8_t>::type:
    success = deSerializeTypedFromBuffer<int8_t>(buffer, size, header, image);
    break;
    case cv::DataType<uint16_t>::type:
    success = deSerializeTypedFromBuffer<uint16_t>(buffer, size, header, image);
    break;
    case cv::DataType<int16_t>::type:
    success = deSerializeTypedFromBuffer<int16_t>(buffer, size, header, image);
    break;
    case cv::DataType<int>::type:
    success = deSerializeTypedFromBuffer<int>(buffer, size, header, image);
    break;
    case cv::DataType<double>::type:
    success = deSerializeTypedFromBuffer<double>(buffer, size, header, image);
    break;
    case cv::DataType<float>::type:
    success = deSerializeTypedFromBuffer<float>(buffer, size, header, image);
    break;
    default:
      LOG(ERROR) << "cv::Mat type " << header.type << " is not supported for "
      << "serialization.";
      success = false;
      break;
  }

  return true;
}

bool serializeToBuffer(const cv::Mat& image, char** buffer, size_t* size) {
  CHECK(image.isContinuous()) << "This method only works if the image is stored "
      "in contiguous memory.";
  CHECK_EQ(image.total(), static_cast<size_t>(image.rows * image.cols * image.channels())) <<
      "Unexpected number of pixels in the image.";
  CHECK_EQ(image.dims, 2) << "This method only works for 2D arrays";
  const char* const matrixData = reinterpret_cast<const char*>(image.data);
  bool success = false;
  // http://docs.opencv.org/modules/core/doc/basic_structures.html#mat-depth
  switch(image.type()) {
    case cv::DataType<uint8_t>::type:
    success = serializeToBuffer<uint8_t>(matrixData, image.rows,
                                         image.cols, image.channels(),
                                         buffer, size);
    case cv::DataType<int8_t>::type:
    success = serializeToBuffer<int8_t>(matrixData, image.rows,
                                        image.cols, image.channels(),
                                        buffer, size);
    break;
    case cv::DataType<uint16_t>::type:
    success = serializeToBuffer<uint16_t>(matrixData, image.rows,
                                          image.cols, image.channels(),
                                          buffer, size);
    break;
    case cv::DataType<int16_t>::type:
    success = serializeToBuffer<int16_t>(matrixData, image.rows,
                                         image.cols, image.channels(),
                                         buffer, size);
    break;
    case cv::DataType<int>::type:
    success = serializeToBuffer<int>(matrixData, image.rows,
                                     image.cols, image.channels(),
                                     buffer, size);
    break;
    case cv::DataType<double>::type:
    success = serializeToBuffer<double>(matrixData, image.rows,
                                        image.cols, image.channels(),
                                        buffer, size);
    break;
    case cv::DataType<float>::type:
    success = serializeToBuffer<float>(matrixData, image.rows,
                                       image.cols, image.channels(),
                                       buffer, size);
    break;
    default:
      LOG(ERROR) << "cv::Mat type " << image.type() << " is not supported for "
      << "serialization.";
      success = false;
      break;
  }
  return success;
}


}  // namespace internal
}  // namespace aslam
