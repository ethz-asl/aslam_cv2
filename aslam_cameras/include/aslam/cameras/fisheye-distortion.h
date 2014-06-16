#ifndef FISHEYE_DISTORTION_H_
#define FISHEYE_DISTORTION_H_

#include <Eigen/Core>
#include <glog/logging.h>
#include <aslam/cameras/distortion.h>

namespace aslam {

class FisheyeDistortion : public aslam::Distortion {
 public:
  explicit FisheyeDistortion(double w) : w_(w) {}
  explicit FisheyeDistortion(const Eigen::VectorXd& params) {
    setParameters(params);
  }

  enum {
    kNumOfParams = 1
  };

  inline static constexpr size_t parameterSize() {
    return kNumOfParams;
  }

  inline int minimalDimensions() const {
    return kNumOfParams;
  }

  void setParameters(const Eigen::VectorXd& parameters) {
    w_ = parameters(0);
  }

  virtual void getParameters(Eigen::VectorXd* params) const {
    CHECK_NOTNULL(params);
    params->resize(1);
    (*params)(0) = w_;
  }

  inline double* getParametersMutable() {
    return &w_;
  }

  void distort(const Eigen::Matrix<double, 2, 1>* point,
               Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;
  void undistort(Eigen::Matrix<double, 2, 1>* point,
                 Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
    CHECK(point);
    CHECK(out_jacobian);
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

  virtual bool operator==(const aslam::Distortion& other) const {
    // TODO(dymczykm) what should we do here if we need to implement method
    // comparing it to base reference, cast?
    return false;
  }

  virtual void distort(const Eigen::Matrix<double, 2, 1>* point) const;

  virtual void distort(const Eigen::Matrix<double, 2, 1>& point,
                       Eigen::Matrix<double, 2, 1>* out_point) const;

  virtual void undistort(Eigen::Matrix<double, 2, 1>* point) const;

  virtual void update(const double*) {
    // TODO(dymczykm) what it is meant for?
    CHECK(false);
  }

  virtual bool distortionParametersValid() const {
    // TODO(dymczykm) any constraints on w? positive?
    return true;
  }

 private:
  double w_;

  static constexpr double kMaxValidAngle = (89.0 * M_PI / 180.0);
};

} // namespace aslam

#include "fisheye-distortion-inl.h"

#endif /* FISHEYE_DISTORTION_H_ */
