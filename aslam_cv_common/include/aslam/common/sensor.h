#ifndef ASLAM_COMMON_SENSOR_H_
#define ASLAM_COMMON_SENSOR_H_

#include <aslam/common/unique-id.h>

namespace aslam {

class Sensor {
  ASLAM_POINTER_TYPEDEFS(Sensor);

  public:
    // Get the sensor id.
    const aslam::SensorId& getId() const { return id_; }

    // Set the camera id.
    void setId(const aslam::SensorId& id) { id_ = id; }

  protected:
    /// The id of this sensor.
    aslam::SensorId id_;

};

}  // namespace aslam

#endif  // ASLAM_COMMON_SENSOR_H_
