#ifndef ASLAM_CAMERAS_DISTORTION_H_
#define ASLAM_CAMERAS_DISTORTION_H_

#include <Eigen/Dense>
#include <aslam/common/macros.h>

  // TODO(slynen) Enable commented out PropertyTree support
//namespace sm {
//class PropertyTree;
//}

namespace aslam {
class Distortion {
 public:
  ASLAM_POINTER_TYPEDEFS(Distortion);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Distortion);

  enum { CLASS_SERIALIZATION_VERSION = 1 };

  // TODO(slynen) Enable commented out PropertyTree support
  //  Distortion(const sm::PropertyTree& property_tree);


//////////////////////////////////////////////////////////////////
/** \name Constructors/destructors and operators
*  @{
*/

 protected:
  Distortion() = delete;

  /// \brief Distortion base constructor. Should only be called by a specialized distortion class ctor.
  ///
  /// @param[in] dist_coeffs Vector containing the distortion parameters.
  ///
  Distortion(const Eigen::VectorXd& dist_coeffs);

 public:
  virtual ~Distortion();

  /// \brief Checks for same distortion type and same parameters.
  ///
  /// @return Same distortion?
  ///
  bool operator==(const Distortion& rhs) const;

/** @} *///////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
/** \name Distort methods: applies the distortion model to a point.
 *  @{
 */

  /// \brief Apply distortion to a point in the normalized image plane
  ///
  /// @param[in,out] y The point in the normalized image plane. After the function, this point is distorted.
  ///
  void distort(Eigen::Vector2d* point) const;

  /// \brief Apply distortion to a point in the normalized image plane
  ///
  /// @param[in] y The point in the normalized image plane.
  /// @param[out] outPoint The distorted point.
  ///
  void distort(const Eigen::Vector2d& point,
               Eigen::Vector2d* out_point) const;

  /// \brief Apply distortion to a point in the normalized image plane
  ///
  /// @param[in,out] y The point in the normalized image plane. After the function, this point is distorted.
  /// @param[out] out_jacobian The Jacobian of the distortion function with respect to small changes in the input point.
  ///
  void distort(Eigen::Vector2d* point,
               Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;

  /// \brief Apply distortion to a point in the normalized image plane using provided distortion coefficients.
  ///        External distortion coefficients can be specified using this function. (Ignores the internally
  ///        stored parameters.
  /// @param[in] dist_coeffs Vector containing the coefficients for the distortion model.
  /// @param[in,out] y The point in the normalized image plane. After the function, this point is distorted.
  /// @param[out] out_jacobian The Jacobian of the distortion function with respect to small changes in the input point.
  ///                   If NULL is passed, the Jacobian calculation is skipped.
  ///
  virtual void distortExternalCoeffs(const Eigen::VectorXd& dist_coeffs,
                                     Eigen::Vector2d* point,
                                     Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  virtual void distortParameterJacobian(const Eigen::VectorXd& dist_coeffs,
                                        const Eigen::Vector2d& point,
                                        Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

/** @} *///////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
/** \name Undistort methods: Removes the modeled distortion effects from a point.
 *  @{
 */

  /// \brief Apply undistortion to recover a point in the normalized image plane.
  ///
  /// @param[in,out] y The distorted point. After the function, this point is in the normalized image plane.
  ///
  void undistort(Eigen::Vector2d* point) const;

  /// \brief Apply undistortion to recover a point in the normalized image plane.
  ///
  /// @param[in] dist_coeffs External distortion coefficients to use.
  /// @param[in,out] y The distorted point. After the function, this point is in the normalized image plane.
  ///
  void undistort(const Eigen::Vector2d& point, Eigen::Vector2d* out_point) const;

  /// \brief Apply undistortion to recover a point in the normalized image plane.
  ///
  /// @param[in,out] y The distorted point. After the function, this point is in the normalized image plane.
  /// @param[out] out_jacobian The Jacobian of the distortion function with respect to small changes in the input point.
  ///             If NULL is passed, the Jacobian calculation is skipped.
  ///
  void undistort(Eigen::Vector2d* point, Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;

  /// \brief Apply undistortion to recover a point in the normalized image plane using provided distortion
  ///        coefficients. External distortion coefficients can be specified using this function. Ignores
  ///        the internally  stored parameters.
  ///
  /// @param[in] dist_coeffs Vector containing the coefficients for the distortion model.
  /// @param[in,out] y The distorted point. After the function, this point is in the normalized image plane.
  /// @param[out] outJy The Jacobian of the distortion function with respect to small changes in the input point.
  ///                   If NULL is passed, the Jacobian calculation is skipped.
  ///
  virtual void undistortExternalCoeffs(const Eigen::VectorXd& dist_coeffs,
                                       Eigen::Vector2d* point,
                                       Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

/** @} *///////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
/** \name Methods to set/get distortion parameters
 *  @{
 */

  /// \brief Set the coefficients for the distortion model.
  ///
  /// @param[in] dist_coeffs Vector containing the coefficients.
  ///
  void setParameters(const Eigen::VectorXd& params);

  /// \brief Get the distortion model coefficients.
  ///
  /// @return Vector containing the coefficients.
  ///
  Eigen::VectorXd getParameters() const;

  /// \brief Check the validity of distortion parameters.
  ///
  /// @param[in] dist_coeffs Vector containing the coefficients.
  ///            Parameters will NOT be stored.
  /// @return If the distortion parameters are valid.
  ///
  virtual bool distortionParametersValid(const Eigen::VectorXd& params) const = 0;
  
 private:
  /** \brief Parameter vector for the distortion model. */
  Eigen::VectorXd dist_coeffs_;

/** @} *///////////////////////////////////////////////////////////////////

};
}  // namespace aslam
#endif  // ASLAM_CAMERAS_DISTORTION_H_
