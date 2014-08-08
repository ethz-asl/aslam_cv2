#ifndef ASLAM_CHANNEL_SERIALIZATION_H_
#define ASLAM_CHANNEL_SERIALIZATION_H_

#include <cstdint>
#include <iostream>

#include <glog/logging.h>
#include <Eigen/Dense>

namespace aslam {
namespace internal {
template<typename T> struct MatrixScalarType;
template<> struct MatrixScalarType<unsigned char>   { enum {value = 0}; };
template<> struct MatrixScalarType<int>    { enum {value = 1}; };
template<> struct MatrixScalarType<float>  { enum {value = 2}; };
template<> struct MatrixScalarType<double> { enum {value = 3}; };

struct HeaderInformation {
  uint32_t rows;
  uint32_t cols;
  uint32_t type;
  size_t size() const;
  bool serializeToBuffer(char* buffer, size_t offset) const;
  bool deSerializeFromBuffer(const char* const bufferIn, size_t offset);
};

template<typename SCALAR>
void makeHeaderInformation(int rows, int cols,
                           HeaderInformation* headerInformation) {
  CHECK_NOTNULL(headerInformation);
  headerInformation->rows = rows;
  headerInformation->cols = cols;
  headerInformation->type = MatrixScalarType<SCALAR>::value;
}

template<typename SCALAR>
bool serializeToString(const char* const matrixData,
                       int rows, int cols,
                       std::string* string) {
  CHECK_NE(rows, -1);
  CHECK_NE(cols, -1);
  CHECK_NOTNULL(string);
  HeaderInformation header;
  makeHeaderInformation<SCALAR>(rows, cols, &header);
  size_t matrixSize = sizeof(SCALAR) * rows * cols;
  size_t totalSize = matrixSize + header.size();

  CHECK_GT(totalSize, 0u);

  string->resize(totalSize);
  char* buffer = &(*string)[0];
  bool success = header.serializeToBuffer(buffer, 0);
  if (!success) {
    LOG(ERROR) << "Failed to serialize header";
    return false;
  }
  size_t offset = header.size();
  memcpy(buffer + offset, matrixData, matrixSize);
  return true;
}

template<typename SCALAR>
bool serializeToBuffer(const char* const matrixData,
                       int rows, int cols,
                       char** buffer, size_t* totalSize) {
  CHECK_NOTNULL(totalSize);
  CHECK_NOTNULL(buffer);
  HeaderInformation header;
  makeHeaderInformation<SCALAR>(rows, cols, &header);
  size_t matrixSize = sizeof(SCALAR) * rows * cols;
  *totalSize = matrixSize + header.size();

  *buffer = new char[*totalSize];
  bool success = header.serializeToBuffer(*buffer, 0);
  if (!success) {
    delete[] *buffer;
    *buffer = nullptr;
    return false;
  }
  size_t offset = header.size();
  memcpy(*buffer + offset, matrixData, matrixSize);
  return true;
}

template<typename SCALAR, int ROWS, int COLS>
bool serializeToBuffer(const Eigen::Matrix<SCALAR, ROWS, COLS>& matrix,
                       char** buffer, size_t* size) {
  const char* const matrixData = reinterpret_cast<const char*>(matrix.data());
  return serializeToBuffer<SCALAR>(matrixData, matrix.rows(), matrix.cols(),
                                   buffer, size);
}

template<typename SCALAR, int ROWS, int COLS>
bool serializeToString(const Eigen::Matrix<SCALAR, ROWS, COLS>& matrix,
                       std::string* string) {
  const char* const matrixData = reinterpret_cast<const char*>(matrix.data());
  return serializeToString<SCALAR>(matrixData, matrix.rows(), matrix.cols(),
                                   string);
}

template<typename SCALAR, int ROWS, int COLS>
bool deSerializeFromBuffer(const char* const buffer, size_t size,
                           Eigen::Matrix<SCALAR, ROWS, COLS>* matrix) {
  CHECK_NOTNULL(matrix);
  HeaderInformation header;
  CHECK_GE(size, header.size());
  bool success = header.deSerializeFromBuffer(buffer, 0);
  if (!success) {
    LOG(ERROR) << "Failed to deserialize header from string: " <<
        std::string(buffer, size);
    return false;
  }
  if (ROWS != Eigen::Dynamic) {
    CHECK_EQ(header.rows, static_cast<uint32_t>(ROWS));
  }
  if (COLS != Eigen::Dynamic) {
    CHECK_EQ(header.cols, static_cast<uint32_t>(COLS));
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
  return deSerializeFromBuffer(string.data(), string.size(), matrix);
}

}  // namespace internal
}  // namespace aslam

#endif  // ASLAM_CHANNEL_SERIALIZATION_H_
