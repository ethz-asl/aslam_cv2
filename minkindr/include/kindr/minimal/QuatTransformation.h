#ifndef KINDR_MINIMAL_QUAT_TRANSFORMATION_H
#define KINDR_MINIMAL_QUAT_TRANSFORMATION_H

#include "RotationQuaternion.h"

namespace kindr {
namespace minimal {

/// \class QuatTransformation
/// \brief A frame transformation built from a quaternion and a point
///
/// This transformation takes points from frame B to frame A, written
/// as \f${}_{A}\mathbf{p} = \mathbf{T}_{AB} {}_{B}\mathbf{p}\f$
///
/// In code, we write:
///
/// \code{.cpp}
/// A_p = T_A_B.transform(B_p);
/// \endcode
///
class QuatTransformation
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef double Scalar;
  typedef Eigen::Vector3d Position;
  typedef RotationQuaternion Rotation;
  typedef Eigen::Matrix4d TransformationMatrix;

  QuatTransformation();

  QuatTransformation(const RotationQuaternion& q_A_B, const Position& A_t_A_B);
  QuatTransformation(const Rotation::Implementation& q_A_B,
                     const Position& A_t_A_B);

  QuatTransformation(const Position& A_t_A_B, const RotationQuaternion& q_A_B);
  QuatTransformation(const Position& A_t_A_B,
                     const Rotation::Implementation& q_A_B);
  
  virtual ~QuatTransformation();

  void setIdentity();

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

  /// \brief check for binary equality
  bool operator==(const QuatTransformation& rhs) const;

 private:
  /// The quaternion that takes vectors from B to A
  ///
  /// \code{.cpp}
  /// A_v = q_A_B_.rotate(B_v);
  /// \endcode
  Rotation q_A_B_;
  /// The vector from the origin of A to the origin of B
  /// expressed in A
  Position A_t_A_B_;
};


std::ostream & operator<<(std::ostream & out, const QuatTransformation& pose);

} // namespace minimal
} // namespace kindr

#endif /* KINDR_MINIMAL_QUAT_TRANSFORMATION_H */
