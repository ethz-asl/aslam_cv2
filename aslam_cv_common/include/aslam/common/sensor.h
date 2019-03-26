#ifndef ASLAM_COMMON_SENSOR_H_
#define ASLAM_COMMON_SENSOR_H_

#include <string>

#include <aslam/common/unique-id.h>

namespace aslam {

class Sensor {
 public:
  ASLAM_POINTER_TYPEDEFS(Sensor);

  Sensor() {};
  Sensor(const aslam::SensorId& id, int sensor_type)
      : id_(id), sensor_type_(sensor_type) {};
  Sensor(const aslam::SensorId& id, int sensor_type, const std::string& topic)
      : id_(id), sensor_type_(sensor_type), topic_(topic) {};
  virtual ~Sensor() = default;

  // Set the sensor id.
  void setId(const aslam::SensorId& id) { id_ = id; };
  void setSensorType(int sensor_type) { sensor_type_ = sensor_type; };
  void setTopic(const std::string& topic) { topic_ = topic; }

  // Get the sensor id.
  const aslam::SensorId& getId() const { return id_; }
  int getSensorType() const { return sensor_type_; }
  const std::string& getTopic() const {return topic_; }

 protected:
  // The id of this sensor.
  aslam::SensorId id_;
  int sensor_type_;
  std::string topic_;

};

}  // namespace aslam

#endif  // ASLAM_COMMON_SENSOR_H_
