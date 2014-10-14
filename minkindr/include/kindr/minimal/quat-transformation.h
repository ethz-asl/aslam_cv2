#ifndef KINDR_MINIMAL_QUAT_TRANSFORMATION_H_
#define KINDR_MINIMAL_QUAT_TRANSFORMATION_H_

#include <kindr/minimal/rotation-quaternion.h>

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
template <typename Scalar>
class QuatTransformationTemplate {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  typedef Eigen::Matrix<Scalar, 3, 1> Position;
  typedef RotationQuaternionTemplate<Scalar> Rotation;
  typedef Eigen::Matrix<Scalar, 4, 4> TransformationMatrix;

  QuatTransformationTemplate();

  QuatTransformationTemplate(
      const RotationQuaternionTemplate<Scalar>& q_A_B, const Position& A_t_A_B);
  QuatTransformationTemplate(
      const typename Rotation::Implementation& q_A_B, const Position& A_t_A_B);

  QuatTransformationTemplate(const Position& A_t_A_B,
                             const Rotation& q_A_B);
  QuatTransformationTemplate(const Position& A_t_A_B,
                             const typename Rotation::Implementation& q_A_B);
  
  QuatTransformationTemplate(const TransformationMatrix& T);

  /// \brief a constructor based on the exponential map
  /// translational part in the first 3 dimensions, 
  /// rotational part in the last 3 dimensions
  QuatTransformationTemplate(const Vector6& x_t_r);

  virtual ~QuatTransformationTemplate();

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
  QuatTransformationTemplate<Scalar> operator*(
      const QuatTransformationTemplate<Scalar>& rhs) const;

  /// \brief transform a point
  Vector3 transform(const Vector3& rhs) const;

  /// \brief transform a point
  Vector4 transform4(const Vector4& rhs) const;

  /// \brief transform a vector (apply only the rotational component)
  Vector3 transformVector(const Vector3& rhs) const;

  /// \brief transform a point by the inverse
  Vector3 inverseTransform(const Vector3& rhs) const;

  /// \brief transform a point by the inverse
  Vector4 inverseTransform4(const Vector4& rhs) const;

  /// \brief transform a vector by the inverse (apply only the rotational
  ///        component)
  Vector3 inverseTransformVector(const Vector3& rhs) const;

  /// \brief get the logarithmic map of the transformation
  Vector6 log() const;

  /// \brief return a copy of the transformation inverted
  QuatTransformationTemplate<Scalar> inverted() const;

  /// \brief check for binary equality
  bool operator==(const QuatTransformationTemplate<Scalar>& rhs) const;

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

typedef QuatTransformationTemplate<double> QuatTransformation;

template<typename Scalar>
std::ostream & operator<<(std::ostream & out,
                          const QuatTransformationTemplate<Scalar>& pose);

} // namespace minimal
} // namespace kindr

#include <kindr/minimal/implementation/quat-transformation-inl.h>

#endif  // KINDR_MINIMAL_QUAT_TRANSFORMATION_H_
