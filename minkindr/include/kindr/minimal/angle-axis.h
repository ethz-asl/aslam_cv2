#ifndef KINDR_MIN_ROTATION_ANGLE_AXIS_H_
#define KINDR_MIN_ROTATION_ANGLE_AXIS_H_

#include <Eigen/Dense>

namespace kindr {
namespace minimal {

template<typename Scalar>
class RotationQuaternionTemplate;

/// \class AngleAxis
/// \brief a minimal implementation of an angle and axis representation of
///        rotation
/// This rotation takes vectors from frame B to frame A, written
/// as \f${}_{A}\mathbf{v} = \mathbf{C}_{AB} {}_{B}\mathbf{v}\f$
///
/// In code, we write:
///
/// \code{.cpp}
/// A_v = C_A_B.rotate(B_v);
/// \endcode
///
template<typename Scalar>
class AngleAxisTemplate {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

  typedef Eigen::AngleAxis<Scalar> Implementation;

  typedef Eigen::Matrix<Scalar, 3, 3> RotationMatrix;

  /// \brief initialize to identity
  AngleAxisTemplate();

  /// \brief initialize from real and imaginary components (real first)
  AngleAxisTemplate(Scalar angle, Scalar v1, Scalar v2, Scalar v3);
  
  /// \brief initialize from real and imaginary components
  AngleAxisTemplate(Scalar real, const Vector3& imaginary);

  /// \brief initialize from an Eigen angleAxis
  AngleAxisTemplate(const Implementation& angleAxis);

  /// \brief initialize from an Eigen angle/axis
  AngleAxisTemplate(const Vector4& angleAxis);

  /// \brief initialize from a rotation matrix
  AngleAxisTemplate(const RotationMatrix& matrix);

  /// \brief initialize from an Eigen quaternion
  AngleAxisTemplate(const RotationQuaternionTemplate<Scalar>& quat);
  
  /// \brief initialize from a angle-scaled axis vector
  AngleAxisTemplate(const Vector3& angleAxis);

  virtual ~AngleAxisTemplate();

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
  AngleAxisTemplate<Scalar> getUnique() const;

  /// \brief set the angle/axis to its unique representation
  AngleAxisTemplate<Scalar>& setUnique();

  /// \brief set the rotation to identity
  AngleAxisTemplate<Scalar>& setIdentity();

  /// \brief get a copy of the rotation inverted.
  AngleAxisTemplate<Scalar> inverted() const;

  /// \brief rotate a vector, v
  Vector3 rotate(const Vector3& v) const;

  /// \brief rotate a vector, v
  Vector4 rotate4(const Vector4& v) const;

  /// \brief rotate a vector, v
  Vector3 inverseRotate(const Vector3& v) const;

  /// \brief rotate a vector, v
  Vector4 inverseRotate4(const Vector4& v) const;

  /// \brief cast to the implementation type
  Implementation& toImplementation();

  /// \brief cast to the implementation type
  const Implementation& toImplementation() const;

  /// \brief get the angle between this and the other rotation
  Scalar getDisparityAngle(const AngleAxisTemplate<Scalar>& rhs) const;

  /// \brief enforce the unit length constraint
  AngleAxisTemplate<Scalar>& normalize();

  /// \brief compose two rotations
  AngleAxisTemplate<Scalar> operator*(
      const AngleAxisTemplate<Scalar>& rhs) const;

  /// \brief assignment operator
  AngleAxisTemplate<Scalar>& operator=(const AngleAxisTemplate<Scalar>& rhs);

  /// \brief get the rotation matrix
  RotationMatrix getRotationMatrix() const;

 private:
  Implementation C_A_B_;
};

typedef AngleAxisTemplate<double> AngleAxis;

template<typename Scalar>
std::ostream& operator<<(std::ostream& out,
                         const AngleAxisTemplate<Scalar>& rhs);

} // namespace minimal
} // namespace kindr

#include <kindr/minimal/implementation/angle-axis-inl.h>

#endif /* KINDR_MIN_ROTATION_ANGLE_AXIS_HPP */
