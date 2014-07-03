#ifndef KINDR_MINIMAL_QUAT_TRANSFORMATION_H
#define KINDR_MINIMAL_QUAT_TRANSFORMATION_H

#include "Position.h"
#include "RotationQuaternion.h"

namespace kindr {
namespace minimal {

class QuatTransformation
{
 public:
  typedef double Scalar;
  typedef Position Position;
  typedef RotationQuaternion Rotation;
  typedef Eigen::Matrix4d TransformationMatrix;

  QuatTransformation();

  QuatTransformation(const RotationQuaternion& rotation, const Position& translation);
  
  virtual ~QuatTransformation();

  /// \brief get the position component
  Position& getPosition();
  
  /// \brief get the position component
  const Position& getPosition() const;

  /// \brief get the rotation component
  Rotation& getRotation();
  
  /// \brief get the rotation component
  const Rotation& getRotation() const;
  
  /// \brief get the transformation matrix
  TransformationMatrix getTransformationMatrix() const;

  /// \brief compose two transformations
  QuatTransformation operator*(const QuatTransformation& rhs) const;

  /// \brief transform a point
  Eigen::Vector3d transform(const Eigen::Vector3d& rhs) const;

  /// \brief transform a point
  Eigen::Vector4d transform4(const Eigen::Vector4d& rhs) const;

  /// \brief transform a vector (apply only the rotational component)
  Eigen::Vector3d transformVector(const Eigen::Vector3d& rhs) const;

  /// \brief transform a point by the inverse
  Eigen::Vector3d inverseTransform(const Eigen::Vector3d& rhs) const;

  /// \brief transform a point by the inverse
  Eigen::Vector4d inverseTransform4(const Eigen::Vector4d& rhs) const;

  /// \brief transform a vector by the inverse (apply only the rotational component)
  Eigen::Vector3d inverseTransformVector(const Eigen::Vector3d& rhs) const;


  /// \brief return a copy of the transformation inverted
  QuatTransformation inverted() const;

  /// \brief invert the transformation
  QuatTransformation& invert();

 private:
  Rotation rotation_;
  Position translation_;
};


std::ostream & operator<<(std::ostream & out, const QuatTransformation& pose);

} // namespace minimal
} // namespace kindr

#endif /* KINDR_MINIMAL_QUAT_TRANSFORMATION_H */
