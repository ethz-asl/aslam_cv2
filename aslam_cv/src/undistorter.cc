#include <aslam/pipeline/undistorter.h>

namespace aslam {

Undistorter::Undistorter() {
}

Undistorter::Undistorter(const std::shared_ptr<Camera>& input_camera,
                         const std::shared_ptr<Camera>& output_camera) :
                           input_camera_(input_camera),
                           output_camera_(output_camera){ }

Undistorter::~Undistorter() { }


}  // namespace aslam
