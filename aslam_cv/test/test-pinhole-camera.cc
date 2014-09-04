#include <aslam/common/entrypoint.h>
#include <aslam/cameras/pinhole-camera.h>
#include <aslam/cameras/fisheye-distortion.h>

TEST(PinHoleCamera, SetGetLineDelay) {
  aslam::PinholeCamera camera;

}

TEST(CameraComparison, TestEquality) {
  using namespace aslam;
	Eigen::VectorXd dvec(1);

	dvec[0] = 0.5;
	Distortion::Ptr distortion_A = std::make_shared<FisheyeDistortion>(dvec);
	Camera::Ptr camera_A = std::make_shared<PinholeCamera>(240, 480, 100, 200, 500, 500, distortion_A);

	dvec[0] = 0.0;
	Distortion::Ptr distortion_B = std::make_shared<FisheyeDistortion>(dvec);
	Camera::Ptr camera_B = std::make_shared<PinholeCamera>(240, 480, 100, 200, 500, 500, distortion_B);

	dvec[0] = 0.5;
	Distortion::Ptr distortion_C = std::make_shared<FisheyeDistortion>(dvec);
	Camera::Ptr camera_C = std::make_shared<PinholeCamera>(11111, 480, 100, 200, 500, 500, distortion_C);

	EXPECT_TRUE( *camera_A == *camera_A );  //same
	EXPECT_FALSE( *camera_A == *camera_B ); //different distortion
	EXPECT_FALSE( *camera_A == *camera_C ); //different intrinsics
}
