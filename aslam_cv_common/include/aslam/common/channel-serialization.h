#ifndef ASLAM_CHANNEL_SERIALIZATION_H_
#define ASLAM_CHANNEL_SERIALIZATION_H_

#include <cstdint>
#include <iostream>
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

template<typename SCALAR>
void makeHeaderInformation(int rows, int cols,
                           HeaderInformation* header_information) {
  CHECK_NOTNULL(header_information);
  header_information->rows = rows;
  header_information->cols = cols;
  header_information->type = MatrixScalarType<SCALAR>::value;
}

template<typename SCALAR>
bool serializeToString(const char* const matrix_data,
                       int rows, int cols,
                       std::string* string) {
  CHECK_NE(rows, -1);
  CHECK_NE(cols, -1);
  CHECK_NOTNULL(string);
  HeaderInformation header;
  makeHeaderInformation<SCALAR>(rows, cols, &header);
  size_t matrix_size = sizeof(SCALAR) * rows * cols;
  size_t total_size = matrix_size + header.size();

  CHECK_GT(total_size, 0u);

  string->resize(total_size);
  char* buffer = &(*string)[0];
  bool success = header.serializeToString(buffer, 0);
  if (!success) {
    LOG(ERROR) << "Failed to serialize header";
    return false;
  }
  size_t offset = header.size();
  memcpy(buffer + offset, matrix_data, matrix_size);
  return true;
}

template<typename SCALAR>
bool serializeToString(const char* const matrix_data,
                       int rows, int cols,
                       char** buffer, size_t* total_size) {
  CHECK_NOTNULL(total_size);
  CHECK_NOTNULL(buffer);
  HeaderInformation header;
  makeHeaderInformation<SCALAR>(rows, cols, &header);
  size_t matrix_size = sizeof(SCALAR) * rows * cols;
  *total_size = matrix_size + header.size();

  *buffer = new char[*total_size];
  bool success = header.serializeToString(*buffer, 0);
  if (!success) {
    delete[] *buffer;
    *buffer = nullptr;
    return false;
  }
  size_t offset = header.size();
  memcpy(*buffer + offset, matrix_data, matrix_size);
  return true;
}

template<typename SCALAR, int ROWS>
bool serializeToString(const Eigen::Matrix<SCALAR, ROWS, Eigen::Dynamic>& matrix,
                       char** buffer, size_t* size) {
  const char* matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, ROWS, matrix.cols(), buffer, size);
}

template<typename SCALAR, int ROWS>
bool serializeToString(const Eigen::Matrix<SCALAR, ROWS, Eigen::Dynamic>& matrix,
                       std::string* string) {
  const char* matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, ROWS, matrix.cols(), string);
}

template<typename SCALAR, int COLS>
bool serializeToString(const Eigen::Matrix<SCALAR, Eigen::Dynamic, COLS>& matrix,
                       char** buffer, size_t* size) {
  const char* matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, matrix.rows(), COLS, buffer, size);
}

template<typename SCALAR, int COLS>
bool serializeToString(const Eigen::Matrix<SCALAR, Eigen::Dynamic, COLS>& matrix,
                       std::string* string) {
  const char* matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, matrix.rows(), COLS, string);
}

template<typename SCALAR>
bool serializeToString(const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>& matrix,
                       char** buffer, size_t* size) {
  const char* matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, matrix.rows(), matrix.cols(), buffer, size);
}

template<typename SCALAR>
bool serializeToString(const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>& matrix,
                       std::string* string) {
  const char* matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, matrix.rows(), matrix.cols(), string);
}

template<typename SCALAR, int ROWS, int COLS>
bool serializeToString(const Eigen::Matrix<SCALAR, ROWS, COLS>& matrix,
                       char** buffer, size_t* size) {
  const char* const matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, ROWS, COLS, buffer, size);
}

template<typename SCALAR, int ROWS, int COLS>
bool serializeToString(const Eigen::Matrix<SCALAR, ROWS, COLS>& matrix,
                       std::string* string) {
  const char* const matrix_data = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrix_data, ROWS, COLS, string);
}

template<typename SCALAR, int ROWS, int COLS>
bool deSerializeFromString(const char* const buffer, size_t size,
                           Eigen::Matrix<SCALAR, ROWS, COLS>* matrix) {
  CHECK_NOTNULL(matrix);
  HeaderInformation header;
  CHECK_GE(size, header.size());
  bool success = header.deSerializeFromString(buffer, 0);
  if (!success) {
    LOG(ERROR) << "Failed to deserialize header from string: " <<
        std::string(buffer, size);
    return false;
  }
  if (ROWS != Eigen::Dynamic) {
    CHECK_EQ(header.rows, ROWS);
  }
  if (COLS != Eigen::Dynamic) {
    CHECK_EQ(header.cols, COLS);
  }
  CHECK_EQ(header.type, MatrixScalarType<SCALAR>::value);

  if (ROWS == Eigen::Dynamic && COLS == Eigen::Dynamic) {
    matrix->resize(header.rows, header.cols);
  } else if (ROWS == Eigen::Dynamic && COLS != Eigen::Dynamic) {
    matrix->resize(header.rows, Eigen::NoChange);
  } else if (ROWS != Eigen::Dynamic && COLS == Eigen::Dynamic) {
    matrix->resize(Eigen::NoChange, header.cols);
  }

  size_t matrix_size = sizeof(SCALAR) * matrix->rows() * matrix->cols();
  size_t total_size = matrix_size + header.size();
  CHECK_EQ(size, total_size);

  memcpy(matrix->data(), buffer + header.size(), matrix_size);
  return true;
}

template<typename SCALAR, int ROWS, int COLS>
bool deSerializeFromString(const std::string& string,
                           Eigen::Matrix<SCALAR, ROWS, COLS>* matrix) {
  CHECK_NOTNULL(matrix);
  return deSerializeFromString(string.data(), string.size(), matrix);
}

}  // namespace internal
}  // namespace aslam

#endif  // ASLAM_CHANNEL_SERIALIZATION_H_
