#ifndef ASLAM_CHANNEL_SERIALIZATION_H_
#define ASLAM_CHANNEL_SERIALIZATION_H_

#include <cstdint>
#include <Eigen/Dense>

namespace aslam {
namespace internal {
template<typename T> struct MatrixScalarType;
template<> struct MatrixScalarType<char>   { enum {value = 0}; };
template<> struct MatrixScalarType<int>    { enum {value = 1}; };
template<> struct MatrixScalarType<float>  { enum {value = 2}; };
template<> struct MatrixScalarType<double> { enum {value = 3}; };

struct HeaderInformation {
  uint32_t rows;
  uint32_t cols;
  uint32_t type;
  size_t size() const;
  bool serializeToString(char* buffer, size_t offset) const;
  bool deSerializeFromString(const char* const buffer_in, size_t offset);
};

template<typename SCALAR, int ROWS, int COLS>
void makeHeaderInformation(const Eigen::Matrix<SCALAR, ROWS, COLS>& matrix,
                           HeaderInformation* header_information) {
  CHECK_NOTNULL(header_information);
  header_information->rows = ROWS;
  header_information->cols = COLS;
  header_information->type = MatrixScalarType<SCALAR>::value;
}

template<typename SCALAR, int ROWS, int COLS>
bool serializeToString(const Eigen::Matrix<SCALAR, ROWS, COLS>& matrix,
                       std::string* string) {
  CHECK_NOTNULL(string);
  HeaderInformation header;
  makeHeaderInformation(matrix, &header);
  size_t matrix_size = sizeof(SCALAR) * ROWS * COLS;
  size_t total_size = matrix_size + header.size();

  string->resize(total_size);
  char* buffer = &(*string)[0];
  bool success = header.serializeToString(buffer, 0);
  if (!success) {
    return false;
  }
  size_t offset = header.size();
  memcpy(buffer + offset, reinterpret_cast<const char*>(matrix.data()),
         matrix_size);
  return true;
}
template<typename SCALAR, int ROWS, int COLS>
bool serializeToString(const Eigen::Matrix<SCALAR, ROWS, COLS>& matrix,
                       char** buffer, size_t* size) {
  CHECK_NOTNULL(size);
  CHECK_NOTNULL(buffer);
  HeaderInformation header;
  makeHeaderInformation(matrix, &header);
  size_t matrix_size = sizeof(SCALAR) * ROWS * COLS;
  size_t total_size = matrix_size + header.size();

  *buffer = new char[total_size];
  bool success = header.serializeToString(*buffer, 0);
  if (!success) {
    delete[] *buffer;
    *buffer = nullptr;
    return false;
  }
  size_t offset = header.size();
  memcpy(*buffer + offset, reinterpret_cast<const char*>(matrix.data()),
         matrix_size);
  return true;
}
template<typename SCALAR, int ROWS, int COLS>
bool deSerializeFromString(const std::string& string,
                           const Eigen::Matrix<SCALAR, ROWS, COLS>* matrix) {
  CHECK_NOTNULL(matrix);

}
template<typename SCALAR, int ROWS, int COLS>
bool deSerializeFromString(const char* const buffer, size_t size,
                           Eigen::Matrix<SCALAR, ROWS, COLS>* matrix) {
  CHECK_NOTNULL(matrix);

}
}  // namespace internal
}  // namespace aslam

#endif  // ASLAM_CHANNEL_SERIALIZATION_H_
