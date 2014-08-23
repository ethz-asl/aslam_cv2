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

  Distortion();
  // TODO(slynen) Enable commented out PropertyTree support
  //  Distortion(const sm::PropertyTree& property_tree);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Distortion);
  virtual ~Distortion();

  ///
  /// \brief Apply distortion to a point in the normalized image plane
  ///
  /// @param[in] params The distortion parameters.
  /// @param[in,out] y The point in the normalized image plane. After the function, this point is distorted.
  ///
  virtual void distort(const Eigen::Map<Eigen::VectorXd>& params, 
                       Eigen::Vector2d* point) const = 0;

  ///
  /// \brief Apply distortion to a point in the normalized image plane
  ///
  /// @param[in] params The distortion parameters.
  /// @param[in] y The point in the normalized image plane.
  /// @param[out] outPoint The distorted point.
  ///
  virtual void distort(const Eigen::Map<Eigen::VectorXd>& params,
                       const Eigen::Vector2d& point,
                       Eigen::Vector2d* out_point) const = 0;
  ///
  /// \brief Apply distortion to a point in the normalized image plane
  ///
  /// @param[in] params The distortion parameters.
  /// @param[in,out] y The point in the normalized image plane. After the function, this point is distorted.
  /// @param[out] outJy The Jacobian of the distortion function with respect to small changes in the input point.
  ///
  virtual void distort(const Eigen::Map<Eigen::VectorXd>& params,
                       Eigen::Vector2d* point,
                       Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  ///
  /// \brief Apply undistortion to recover a point in the normalized image plane.
  ///
  /// @param[in] params The distortion parameters.
  /// @param[in,out] y The distorted point. After the function, this point is in the normalized image plane.
  ///
  virtual void undistort(const Eigen::Map<Eigen::VectorXd>& params,
                         Eigen::Vector2d* point) const = 0;

  ///
  /// \brief Apply undistortion to recover a point in the normalized image plane.
  ///
  /// @param[in] params The distortion parameters.
  /// @param[in,out] y The distorted point. After the function, this point is in the normalized image plane.
  /// @param[out] outJy The Jacobian of the undistortion function with respect to small changes in the input point.
  ///
  virtual void undistort(
      const Eigen::Map<Eigen::VectorXd>& params,
      Eigen::Vector2d* point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  ///
  /// \brief Apply distortion to the point and provide the Jacobian of the
  ///distortion with respect to small changes in the distortion parameters
  ///
  /// @param[in] params The distortion parameters.
  /// @param[in] point the point in the normalized image plane.
  /// @param[out] outJd  the Jacobian of the distortion with respect to small changes
  ///                    in the distortion parameters.
  ///
  virtual void distortParameterJacobian(
      const Eigen::Map<Eigen::VectorXd>& params,
      const Eigen::Vector2d& point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;


  ///
  /// \brief Getter for the number of distortion parameters.
  ///
  /// @return The 2-element of distortion parameters.
  ///
  virtual size_t getParameterSize() const = 0;

  ///
  /// \brief Getter for the validity of distortion parameters.
  ///
  /// @return If the distortion parameters are valid.
  ///
  virtual bool distortionParametersValid(const Eigen::Map<Eigen::VectorXd>& params) const = 0;
};
}  // namespace aslam
#endif  // ASLAM_CAMERAS_DISTORTION_H_
