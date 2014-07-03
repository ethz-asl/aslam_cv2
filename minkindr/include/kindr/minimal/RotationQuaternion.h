#ifndef KINDR_MIN_ROTATION_QUATERNION_HPP
#define KINDR_MIN_ROTATION_QUATERNION_HPP

#include <Eigen/Dense>

namespace kindr {
namespace minimal {

class AngleAxis;

/// \class RotationQuaternion
/// \brief a minimal implementation of a passive Hamiltonian rotation (unit-length) quaternion
class RotationQuaternion
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  typedef double Scalar;
  
  typedef Eigen::Vector3d Imaginary;
  
  typedef Eigen::Vector4d Vector4;

  typedef Eigen::Quaternion<double> Implementation;

  typedef Eigen::Matrix3d RotationMatrix;

  /// \brief initialize to identity
  RotationQuaternion();

  /// \brief initialize from real and imaginary components (real first)
  RotationQuaternion(const Vector4& quat);
  

  /// \brief initialize from real and imaginary components (real first)
  RotationQuaternion(Scalar w, Scalar x, Scalar y, Scalar z);
  
  /// \brief initialize from real and imaginary components
  RotationQuaternion(Scalar real, const Eigen::Vector3d& imaginary);

  /// \brief initialize from an Eigen quaternion
  RotationQuaternion(const Implementation& quaternion);

  /// \brief initialize from a rotation matrix
  RotationQuaternion(const RotationMatrix& matrix);

  /// \brief initialize from an AngleAxis
  RotationQuaternion(const AngleAxis& angleAxis);
  
  virtual ~RotationQuaternion();

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
  RotationQuaternion getUnique() const;

  /// \brief set the quaternion to its unique representation
  RotationQuaternion& setUnique();

  /// \brief set the quaternion to identity
  RotationQuaternion& setIdentity();

  /// \brief invert the quaternion
  RotationQuaternion& invert();

  /// \brief get a copy of the quaternion inverted.
  RotationQuaternion inverted() const;

  /// \brief conjugate the quaternion
  RotationQuaternion& conjugate();

  /// \brief get a copy of the conjugate of the quaternion.
  RotationQuaternion conjugated() const;

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

  /// \brief get the norm of the quaternion
  Scalar norm() const;

  /// \brief get the squared norm of the quaternion
  Scalar squaredNorm() const;

  /// \brief get the angle between this and the other quaternion
  Scalar getDisparityAngle(const RotationQuaternion& rhs) const;

  /// \brief enforce the unit length constraint
  RotationQuaternion& fix();

  /// \brief compose two quaternions
  RotationQuaternion operator*(const RotationQuaternion& rhs) const;

  /// \brief assignment operator
  RotationQuaternion& operator=(const RotationQuaternion& rhs);

  /// \brief get the rotation matrix
  RotationMatrix getRotationMatrix() const;

 private:
  Implementation quaternion_;
  
};

std::ostream& operator<<(std::ostream& out, const RotationQuaternion& rhs);

} // namespace minimal
} // namespace kindr


#endif /* KINDR_MIN_ROTATION_QUATERNION_HPP */
