#ifndef FISHEYE_DISTORTION_H_
#define FISHEYE_DISTORTION_H_

#include <Eigen/Core>
#include <glog/logging.h>
#include "distortion.h"

namespace aslam {

class FisheyeDistortion : public aslam::Distortion {
 private:
  enum { kNumOfParams = 1 };

 public:
  enum { CLASS_SERIALIZATION_VERSION = 1 };

  explicit FisheyeDistortion() { }

  virtual size_t getParameterSize() const {
    return kNumOfParams;
  }

  inline static constexpr size_t parameterCount() {
    return kNumOfParams;
  }

  bool operator==(const Distortion& rhs) const;

  void distort(const Eigen::Map<const Eigen::VectorXd>& params,
               Eigen::Matrix<double, 2, 1>* point,
               Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;

  void undistort(const Eigen::Map<const Eigen::VectorXd>& /* params */,
                 Eigen::Matrix<double, 2, 1>* point,
                 Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
    CHECK(point);
    CHECK(out_jacobian);
    // TODO(dymczykm) to be implemented at some point
    CHECK(false);
  }

  // templated versions, e.g. for ceres autodiff
  template <typename ScalarType>
  void distort(const Eigen::Map<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>>& params,
               const Eigen::Matrix<ScalarType, 2, 1>& point,
               Eigen::Matrix<ScalarType, 2, 1>* out_point) const;

  void distortParameterJacobian(
      const Eigen::Map<const Eigen::VectorXd>& params,
      const Eigen::Matrix<double, 2, 1>& point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;

  virtual void distort(const Eigen::Map<const Eigen::VectorXd>& params,
                       Eigen::Matrix<double, 2, 1>* point) const;

  virtual void distort(const Eigen::Map<const Eigen::VectorXd>& params,
                       const Eigen::Matrix<double, 2, 1>& point,
                       Eigen::Matrix<double, 2, 1>* out_point) const;

  virtual void undistort(const Eigen::Map<const Eigen::VectorXd>& params,
                         Eigen::Matrix<double, 2, 1>* point) const;

  virtual bool distortionParametersValid(const Eigen::Map<const Eigen::VectorXd>& params) const {
    CHECK_EQ(params.size(), 1)
        << "Invalid number of distortion coefficients (found "
        << params.size() << ", expected 1).";

    // Expect w to have sane magnitude.
    double w = params(0);
    bool valid = std::abs(w) < 1e-16 || (w >= kMinValidW && w <= kMaxValidW);
    LOG_IF(INFO, !valid) << "Invalid w parameter: " << w << ", expected w in [" << kMinValidW
        << ", " << kMaxValidW << "].";
    return valid;
  }

 private:
  static constexpr double kMaxValidAngle = (89.0 * M_PI / 180.0);
  static constexpr double kMinValidW = 0.5;
  static constexpr double kMaxValidW = 1.5;
};

} // namespace aslam

#include "fisheye-distortion-inl.h"

#endif /* FISHEYE_DISTORTION_H_ */
