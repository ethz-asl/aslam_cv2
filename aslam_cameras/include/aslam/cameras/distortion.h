#ifndef ASLAM_CAMERAS_DISTORTION_H_
#define ASLAM_CAMERAS_DISTORTION_H_

#include <Eigen/Dense>
#include <aslam/common/macros.h>

//namespace sm {
//class PropertyTree;
//}

namespace aslam {
class Distortion {
 public:
  ASLAM_POINTER_TYPEDEFS(Distortion);

  Distortion();
//  Distortion(const sm::PropertyTree& property_tree);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Distortion);
  virtual ~Distortion();
  virtual bool operator==(const Distortion& other) const = 0;

  /**
   * \brief Apply distortion to a point in the normalized image plane
   *
   * @param y The point in the normalized image plane. After the function, this point is distorted.
   */
  virtual void distort(const Eigen::Matrix<double, 2, 1>* y) const = 0;

  /**
   * \brief Apply distortion to a point in the normalized image plane
   *
   * @param y The point in the normalized image plane.
   * @param outPoint The distorted point.
   */
  virtual void distort(const Eigen::Matrix<double, 2, 1>& point,
               Eigen::Matrix<double, 2, 1>* out_point) const = 0;
  /**
   * \brief Apply distortion to a point in the normalized image plane
   *
   * @param y The point in the normalized image plane. After the function, this point is distorted.
   * @param outJy The Jacobian of the distortion function with respect to small changes in the input point.
   */
  virtual void distort(const Eigen::Matrix<double, 2, 1>* point,
               Eigen::Matrix<double, 2, Eigen::Dynamic>* outJy) const = 0;

  /**
   * \brief Apply undistortion to recover a point in the normalized image plane.
   *
   * @param y The distorted point. After the function, this point is in the normalized image plane.
   */
  virtual void undistort(Eigen::Matrix<double, 2, 1>* y) const = 0;

  /**
   * \brief Apply undistortion to recover a point in the normalized image plane.
   *
   * @param y The distorted point. After the function, this point is in the normalized image plane.
   * @param outJy The Jacobian of the undistortion function with respect to small changes in the input point.
   */
  virtual void undistort(Eigen::Matrix<double, 2, 1>* y,
                 Eigen::Matrix<double, 2, Eigen::Dynamic>* outJy) const = 0;

  /**
   * \brief Apply distortion to the point and provide the Jacobian of the
   * distortion with respect to small changes in the distortion parameters
   *
   * @param imageY the point in the normalized image plane.
   * @param outJd  the Jacobian of the distortion with respect to small changes
   * in the distortion parameters.
   */
  virtual void distortParameterJacobian(
      Eigen::Matrix<double, 2, 1>* imageY,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* outJd) const = 0;

  /**
   * \brief A function for compatibility with the aslam backend.
   * This implements an update of the distortion parameter.
   *
   * @param v A double array representing the update vector.
   */
  virtual void update(const double* v) = 0;

  /**
   * \brief A function for compatibility with the aslam backend.
   *
   * @param v The number of parameters expected by the update equation.
   * This should also define the number of columns in the matrix returned by distortParameterJacobian.
   */
  virtual int minimalDimensions() const = 0;

  /**
   * \brief A function for compatibility with the aslam backend.
   *
   * @param parameters This vector is resized and filled with parameters representing the full state of the distortion.
   */
  virtual void getParameters(Eigen::VectorXd* parameters) const = 0;

  /**
   * \brief A function for compatibility with the aslam backend.
   *
   * @param parameters The full state of the distortion class is set from the vector of parameters.
   */
  virtual void setParameters(const Eigen::VectorXd& parameters) = 0;

  /**
   * \brief A function for compatibility with the ceres solver.
   *
   * @return The underlying raw pointer to the distortion parameters.
   */
  virtual double* getParametersMutable() = 0;

  /**
   * \brief Getter for the number of distortion parameters.
   *
   * @return The number of distortion parameters.
   */
  // TODO(dymczykm) must be constexpr
  //virtual size_t parameterSize() const = 0;

  /**
   * \brief Getter for the validity of distortion parameters.
   *
   * @return If the distortion parameters are valid.
   */
  virtual bool distortionParametersValid() const = 0;
};
}  // namespace aslam
#endif  // ASLAM_CAMERAS_DISTORTION_H_
