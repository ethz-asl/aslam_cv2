#ifndef KINDR_MIN_ROTATION_QUATERNION_H_
#define KINDR_MIN_ROTATION_QUATERNION_H_

#include <Eigen/Dense>

namespace kindr {
namespace minimal {

template<typename Scalar>
class AngleAxisTemplate;

/// \class RotationQuaternion
/// \brief a minimal implementation of a passive Hamiltonian rotation
///        (unit-length) quaternion
///
/// This rotation takes vectors from frame B to frame A, written
/// as \f${}_{A}\mathbf{v} = \mathbf{C}_{AB} {}_{B}\mathbf{v}\f$
///
/// In code, we write:
///
/// \code{.cpp}
/// A_v = q_A_B.rotate(B_v);
/// \endcode
///
template<typename Scalar>
class RotationQuaternionTemplate {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;

  typedef Vector3 Imaginary;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

  typedef Eigen::Quaternion<Scalar> Implementation;

  typedef Eigen::Matrix<Scalar, 3, 3> RotationMatrix;

  /// \brief initialize to identity
  RotationQuaternionTemplate();

  /// \brief initialize from real and imaginary components (real first)
  RotationQuaternionTemplate(const Vector4& quat);

  /// \brief initialize from angle scaled axis
  RotationQuaternionTemplate(const Vector3& angle_scaled_axis);

  /// \brief initialize from real and imaginary components (real first)
  RotationQuaternionTemplate(Scalar w, Scalar x, Scalar y, Scalar z);

  /// \brief initialize from real and imaginary components
  RotationQuaternionTemplate(Scalar real, const Imaginary& imaginary);

  /// \brief initialize from an Eigen quaternion
  RotationQuaternionTemplate(const Implementation& quaternion);

  /// \brief initialize from a rotation matrix
  RotationQuaternionTemplate(const RotationMatrix& matrix);

  /// \brief initialize from an AngleAxis
  RotationQuaternionTemplate(const AngleAxisTemplate<Scalar>& angleAxis);

  virtual ~RotationQuaternionTemplate();

  /// \brief the real component of the quaternion
  Scalar w() const;
  /// \brief the first imaginary component of the quaternion
  Scalar x() const;
  /// \brief the second imaginary component of the quaternion
  Scalar y() const;
  /// \brief the third imaginary component of the quaternion
  Scalar z() const;

  /// \brief the imaginary components of the quaterion.
  Imaginary imaginary() const;

  /// \brief get the components of the quaternion as a vector (real first)
  Vector4 vector() const;

  /// \brief set the quaternion by its values (real, imaginary)
  void setValues(Scalar w, Scalar x, Scalar y, Scalar z);

  /// \brief set the quaternion by its real and imaginary parts
  void setParts(Scalar real, const Imaginary& imag);

  /// \brief get a copy of the representation that is unique
  RotationQuaternionTemplate<Scalar> getUnique() const;

  /// \brief set the quaternion to its unique representation
  RotationQuaternionTemplate<Scalar>& setUnique();

  /// \brief set the quaternion to identity
  RotationQuaternionTemplate<Scalar>& setIdentity();

  /// \brief get a copy of the quaternion inverted.
  RotationQuaternionTemplate<Scalar> inverted() const;

  /// \brief get a copy of the conjugate of the quaternion.
  RotationQuaternionTemplate<Scalar> conjugated() const;

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

  /// \brief get the norm of the quaternion
  Scalar norm() const;

  /// \brief get the squared norm of the quaternion
  Scalar squaredNorm() const;

  /// \brief get the angle between this and the other quaternion
  Scalar getDisparityAngle(const RotationQuaternionTemplate<Scalar>& rhs) const;

  /// \brief enforce the unit length constraint
  RotationQuaternionTemplate<Scalar>& normalize();

  /// \brief compose two quaternions
  RotationQuaternionTemplate<Scalar> operator*(
      const RotationQuaternionTemplate<Scalar>& rhs) const;

  /// \brief assignment operator
  RotationQuaternionTemplate<Scalar>& operator=(
      const RotationQuaternionTemplate<Scalar>& rhs);

  /// \brief get the rotation matrix
  RotationMatrix getRotationMatrix() const;

  /// \brief check for binary equality
  bool operator==(const RotationQuaternionTemplate<Scalar>& rhs) const {
    return vector() == rhs.vector();
  }

 private:
  Implementation q_A_B_;

};

typedef RotationQuaternionTemplate<double> RotationQuaternion;

template<typename Scalar>
std::ostream& operator<<(std::ostream& out,
                         const RotationQuaternionTemplate<Scalar>& rhs);

}  // namespace minimal
}  // namespace kindr

#include <kindr/minimal/implementation/rotation-quaternion-inl.h>

#endif  // KINDR_MIN_ROTATION_QUATERNION_H_
