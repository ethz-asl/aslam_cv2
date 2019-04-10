#ifndef ASLAM_COMMON_SENSOR_H_
#define ASLAM_COMMON_SENSOR_H_

#include <string>

#include <aslam/common/macros.h>
#include <aslam/common/unique-id.h>
#include <aslam/common/yaml-serialization.h>

namespace aslam {
enum SensorType : int {
  kNCamera,
  kCamera
};

constexpr char kYamlFieldNameId[] = "id";
// TODO(smauq): Fix
//constexpr char kYamlFieldNameSensorType[] = "sensor_type";
constexpr char kYamlFieldNameHardwareId[] = "topic";

class Sensor {
 public:
  ASLAM_POINTER_TYPEDEFS(Sensor);

  Sensor() {};
  explicit Sensor(int sensor_type);
  explicit Sensor(const SensorId& id);
  Sensor(const SensorId& id, int sensor_type);
  Sensor(const SensorId& id, int sensor_type, const std::string& topic);

  virtual ~Sensor() = default;

  Sensor(const Sensor& other)
      : id_(other.id_),
        sensor_type_(other.sensor_type_),
        topic_(other.topic_) {}
  void operator=(const Sensor& other) = delete;

  // Set the sensor id.
  void setId(const SensorId& id);
  void setSensorType(int sensor_type) { sensor_type_ = sensor_type; };
  void setTopic(const std::string& topic) { topic_ = topic; }

  // Get the sensor id.
  const SensorId& getId() const { return id_; }
  int getSensorType() const { return sensor_type_; }
  const std::string& getTopic() const {return topic_; }

  bool isValid() const;

  bool deserialize(const YAML::Node& sensor_node);
  void serialize(YAML::Node* sensor_node_ptr) const;

 private:
  void setRandom();
  virtual bool isValidImpl() const = 0;
  virtual void setRandomImpl() = 0;

  virtual bool loadFromYamlNodeImpl(const YAML::Node& sensor_node) = 0;
  virtual void saveToYamlNodeImpl(YAML::Node* sensor_node) const = 0;

 protected:
  // The id of this sensor.
  SensorId id_;
  int sensor_type_;
  std::string topic_;

};
}  // namespace aslam

#endif  // ASLAM_COMMON_SENSOR_H_
