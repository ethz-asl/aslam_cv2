#include <aslam/cameras/ncamera-factory.h>

#include <string>
#include <vector>


#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/memory.h>

namespace aslam {

  aslam::NCamera::Ptr createPlanar4CameraRig() {
    std::vector<aslam::Camera::Ptr> cameras;
    cameras.push_back(aslam::PinholeCamera::createTestCamera());
    cameras.push_back(aslam::PinholeCamera::createTestCamera());
    cameras.push_back(aslam::PinholeCamera::createTestCamera());
    cameras.push_back(aslam::PinholeCamera::createTestCamera());
    aslam::NCameraId rig_id;
    rig_id.randomize();
    // This defines an artificial camera system similar to the one on the V-Charge or JanETH car.
    aslam::Position3D t_B_C0(2.0, 0.0, 0.0);
    Eigen::Matrix3d R_B_C0 = Eigen::Matrix3d::Zero();
    R_B_C0(1, 0) = -1.0;
    R_B_C0(2, 1) = -1.0;
    R_B_C0(0, 2) = 1.0;
    aslam::Quaternion q_B_C0(R_B_C0);

    aslam::Position3D t_B_C1(0.0, 1.0, 0.0);
    Eigen::Matrix3d R_B_C1 = Eigen::Matrix3d::Zero();
    R_B_C1(0, 0) = 1.0;
    R_B_C1(2, 1) = -1.0;
    R_B_C1(1, 2) = 1.0;
    aslam::Quaternion q_B_C1(R_B_C1);

    aslam::Position3D t_B_C2(-1.0, 0.0, 0.0);
    Eigen::Matrix3d R_B_C2 = Eigen::Matrix3d::Zero();
    R_B_C2(1, 0) = 1.0;
    R_B_C2(2, 1) = -1.0;
    R_B_C2(0, 2) = -1.0;
    aslam::Quaternion q_B_C2(R_B_C2);

    aslam::Position3D t_B_C3(0.0, -1.0, 0.0);
    Eigen::Matrix3d R_B_C3 = Eigen::Matrix3d::Zero();
    R_B_C3(0, 0) = -1.0;
    R_B_C3(2, 1) = -1.0;
    R_B_C3(1, 2) = -1.0;
    aslam::Quaternion q_B_C3(R_B_C3);

    aslam::NCamera::TransformationVector rig_transformations;
    rig_transformations.emplace_back(q_B_C0.inverted(), -t_B_C0);
    rig_transformations.emplace_back(q_B_C1.inverted(), -t_B_C1);
    rig_transformations.emplace_back(q_B_C2.inverted(), -t_B_C2);
    rig_transformations.emplace_back(q_B_C3.inverted(), -t_B_C3);

    std::string label = "Artificial Planar 4-Pinhole-Camera-Rig";

    return aslam::aligned_shared<aslam::NCamera>(rig_id, rig_transformations, cameras, label);
  }

  aslam::NCamera::Ptr createSingleCameraRig() {
    std::vector<aslam::Camera::Ptr> cameras;
    cameras.push_back(aslam::PinholeCamera::createTestCamera());
    aslam::NCameraId rig_id;
    rig_id.randomize();
    // This defines an artificial camera system similar to the one on the V-Charge or JanETH car.
    aslam::Position3D t_B_C0(2.0, 0.0, 0.0);
    Eigen::Matrix3d R_B_C0 = Eigen::Matrix3d::Zero();
    R_B_C0(1, 0) = -1.0;
    R_B_C0(2, 1) = -1.0;
    R_B_C0(0, 2) = 1.0;
    aslam::Quaternion q_B_C0(R_B_C0);

    aslam::NCamera::TransformationVector rig_transformations;
    rig_transformations.emplace_back(q_B_C0.inverted(), -t_B_C0);

    std::string label = "Artificial Planar 1-Pinhole-Camera-Rig";

    return aslam::aligned_shared<aslam::NCamera>(rig_id, rig_transformations, cameras, label);
  }

}  // namespace aslam
