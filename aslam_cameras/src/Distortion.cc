#include <aslam/cameras/Distortion.h>
#include <glog/logging.h>
#include <sm/PropertyTree.h>
namespace aslam {
namespace cameras {
Distortion::Distortion() { }
Distortion::Distortion(const sm::PropertyTree& property_tree) { }
Distortion::~Distortion() { }
virtual bool operator==(const Distortion& other) const {
  return true;
}
}  // namespace cameras
}  // namespace aslam
