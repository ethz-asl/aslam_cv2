#ifndef KINDR_MIN_ROTATION_ANGLE_AXIS_HPP
#define KINDR_MIN_ROTATION_ANGLE_AXIS_HPP

#include <Eigen/Dense>

namespace kindr {
namespace minimal {

class RotationQuaternion;

/// \class AngleAxis
/// \brief a minimal implementation of an angle and axis representation of rotation
/// This rotation takes vectors from frame B to frame A, written
/// as \f${}_{A}\mathbf{v} = \mathbf{C}_{AB} {}_{B}\mathbf{v}\f$
///
/// In code, we write:
///
/// \code{.cpp}
/// A_v = C_A_B.rotate(B_v);
/// \endcode
///
class AngleAxis
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  typedef double Scalar;
  
  typedef Eigen::Vector3d Vector3;
  
  typedef Eigen::Vector4d Vector4;

  typedef Eigen::AngleAxis<double> Implementation;

  typedef Eigen::Matrix3d RotationMatrix;

  /// \brief initialize to identity
  AngleAxis();

  /// \brief initialize from real and imaginary components (real first)
  AngleAxis(Scalar angle, Scalar v1, Scalar v2, Scalar v3);
  
  /// \brief initialize from real and imaginary components
  AngleAxis(Scalar real, const Vector3& imaginary);

  /// \brief initialize from an Eigen angleAxis
  AngleAxis(const Implementation& angleAxis);

  /// \brief initialize from an Eigen angle/axis
  AngleAxis(const Vector4& angleAxis);

  /// \brief initialize from a rotation matrix
  AngleAxis(const RotationMatrix& matrix);

  /// \brief initialize from an Eigen quaternion
  AngleAxis(const RotationQuaternion& quat);
  
  virtual ~AngleAxis();

  /// \brief Returns the rotation angle.
  Scalar angle() const;

  /// \brief Sets the rotation angle.
  void setAngle(Scalar angle);

  /// \brief Returns the rotation axis.
  const Vector3& axis() const;

  /// \brief Sets the rotation axis.
  void setAxis(const Vector3& axis);

  /// \brief Sets the rotation axis.
  void setAxis(Scalar v1, Scalar v2, Scalar v3);

  /// \brief get the components of the angle/axis as a vector (angle first)
  Vector4 vector() const;

  /// \brief get a copy of the representation that is unique
  AngleAxis getUnique() const;

  /// \brief set the angle/axis to its unique representation
  AngleAxis& setUnique();

  /// \brief set the rotation to identity
  AngleAxis& setIdentity();

  /// \brief invert the rotation
  AngleAxis& invert();

  /// \brief get a copy of the rotation inverted.
  AngleAxis inverted() const;

  /// \brief rotate a vector, v
  Eigen::Vector3d rotate(const Eigen::Vector3d& v) const;

  /// \brief rotate a vector, v
  Eigen::Vector4d rotate4(const Eigen::Vector4d& v) const;

  /// \brief rotate a vector, v
  Eigen::Vector3d inverseRotate(const Eigen::Vector3d& v) const;

  /// \brief rotate a vector, v
  Eigen::Vector4d inverseRotate4(const Eigen::Vector4d& v) const;

  /// \brief cast to the implementation type
  Implementation& toImplementation();

  /// \brief cast to the implementation type
  const Implementation& toImplementation() const;

  /// \brief get the angle between this and the other rotation
  Scalar getDisparityAngle(const AngleAxis& rhs) const;

  /// \brief enforce the unit length constraint
  AngleAxis& fix();

  /// \brief compose two rotations
  AngleAxis operator*(const AngleAxis& rhs) const;

  /// \brief assignment operator
  AngleAxis& operator=(const AngleAxis& rhs);

  /// \brief get the rotation matrix
  RotationMatrix getRotationMatrix() const;

 private:
  Implementation C_A_B_;
  
};

std::ostream& operator<<(std::ostream& out, const AngleAxis& rhs);

} // namespace minimal
} // namespace kindr


#endif /* KINDR_MIN_ROTATION_ANGLE_AXIS_HPP */
