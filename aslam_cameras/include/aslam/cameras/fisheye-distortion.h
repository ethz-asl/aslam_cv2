#ifndef FISHEYE_DISTORTION_H_
#define FISHEYE_DISTORTION_H_

#include <Eigen/Core>
#include <glog/logging.h>
#include <aslam/cameras/distortion.h>

namespace aslam {

class FisheyeDistortion : public aslam::Distortion {
 private:
  enum {
    kNumOfParams = 1
  };

 public:
  explicit FisheyeDistortion(double w) {
    params_.resize(1, 1);
    params_(0) = w;
  }

  explicit FisheyeDistortion(const Eigen::VectorXd& params) {
    setParameters(params);
  }

  inline static constexpr size_t parameterSize() {
    return kNumOfParams;
  }

  inline int minimalDimensions() const {
    return kNumOfParams;
  }

  void setParameters(const Eigen::VectorXd& parameters) {
    CHECK_EQ(parameters.rows(), 1);
    params_ = parameters;
  }

  virtual void getParameters(Eigen::VectorXd* parameters) const {
    CHECK_NOTNULL(parameters);
    *parameters = params_;
  }

  virtual Eigen::VectorXd& getParametersMutable() {
    return params_;
  }

  virtual double* getParameterMutablePtr() {
    return params_.data();
  }

  void distort(Eigen::Matrix<double, 2, 1>* point,
               Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;
  void undistort(Eigen::Matrix<double, 2, 1>* point,
                 Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
    CHECK(point);
    CHECK(out_jacobian);
    // TODO(dymczykm) to be implemented at some point
    CHECK(false);
  }

  // templated versions, e.g. for ceres autodiff
  template <typename ScalarType>
  void distort(const Eigen::Matrix<ScalarType, 2, 1>& point,
               const Eigen::Matrix<ScalarType, kNumOfParams, 1>& params,
               Eigen::Matrix<ScalarType, 2, 1>* out_point) const;

  void distortParameterJacobian(
      const Eigen::Matrix<double, 2, 1>& point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;

  virtual bool operator==(const aslam::Distortion& other) const;

  virtual void distort(Eigen::Matrix<double, 2, 1>* point) const;

  virtual void distort(const Eigen::Matrix<double, 2, 1>& point,
                       Eigen::Matrix<double, 2, 1>* out_point) const;

  virtual void undistort(Eigen::Matrix<double, 2, 1>* point) const;

  virtual void update(const double* d_w) {
    CHECK_NOTNULL(d_w);
    params_(0) += d_w[0];
  }

  virtual bool distortionParametersValid() const {
    CHECK_EQ(params_.rows(), 1)
        << "Invalid number of distortion coefficients (found "
        << params_.rows() << ", expected 1).";

    // Expect w to have sane magnitude.
    double w = params_(0);
    CHECK(w >= kMinValidW && w <= kMaxValidW)
        << "Invalid w parameter: " << w << ", expected w in [" << kMinValidW
        << ", " << kMaxValidW << "].";
    return true;
  }

 private:
  Eigen::VectorXd params_;

  static constexpr double kMaxValidAngle = (89.0 * M_PI / 180.0);
  static constexpr double kMinValidW = 0.5;
  static constexpr double kMaxValidW = 1.5;
};

} // namespace aslam

#include "fisheye-distortion-inl.h"

#endif /* FISHEYE_DISTORTION_H_ */
