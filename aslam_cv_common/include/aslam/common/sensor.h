#ifndef ASLAM_COMMON_SENSOR_H_
#define ASLAM_COMMON_SENSOR_H_

#include <string>

#include "aslam/common/macros.h"
#include "aslam/common/pose-types.h"
#include "aslam/common/unique-id.h"
#include "aslam/common/yaml-file-serialization.h"
#include "aslam/common/yaml-serialization.h"

namespace aslam {

enum SensorType : uint8_t { kUnknown, kNCamera, kCamera };

constexpr const char kNCameraIdentifier[] = "NCAMERA";
constexpr const char kCameraIdentifier[] = "CAMERA";

constexpr const char kYamlFieldNameId[] = "id";
constexpr const char kYamlFieldNameSensorType[] = "sensor_type";
constexpr const char kYamlFieldNameTopic[] = "topic";
constexpr const char kYamlFieldNameDescription[] = "description";
constexpr const char kYamlFieldNameBaseSensorId[] = "base_sensor_id";

class Sensor : public YamlFileSerializable {
 public:
  ASLAM_POINTER_TYPEDEFS(Sensor);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sensor();
  explicit Sensor(const SensorId& id);
  explicit Sensor(const SensorId& id, const std::string& topic);
  explicit Sensor(
      const SensorId& id, const std::string& topic,
      const std::string& description);

  virtual ~Sensor() = default;

  Sensor(const Sensor& other)
      : id_(other.id_),
        topic_(other.topic_),
        description_(other.description_),
        base_sensor_id_(other.base_sensor_id_),
        T_B_S_(other.T_B_S_) {}
  void operator=(const Sensor& other) {
    id_ = other.id_;
    topic_ = other.topic_;
    description_ = other.description_;
    base_sensor_id_ = other.base_sensor_id_;
    T_B_S_ = other.T_B_S_;
  }

  bool operator==(const Sensor& other) const;
  bool operator!=(const Sensor& other) const;
  bool isEqual(const Sensor& other, const bool verbose = false) const;

  virtual Sensor::Ptr cloneAsSensor() const = 0;

  // Set and get the sensor id.
  void setId(const SensorId& id) {
    CHECK(id.isValid());
    id_ = id;
  }
  const SensorId& getId() const {
    CHECK(id_.isValid());
    return id_;
  }

  // Set and get the topic
  void setTopic(const std::string& topic) {
    topic_ = topic;
  }
  const std::string& getTopic() const {
    return topic_;
  }

  // Set and get the description
  void setDescription(const std::string& description) {
    description_ = description;
  }
  const std::string& getDescription() const {
    return description_;
  }

  // Set and get the transformation to the base sensor
  void set_T_B_S(const Transformation& T_B_S) {
    T_B_S_ = T_B_S;
  }
  const Transformation& get_T_B_S() const {
    return T_B_S_;
  }

  void setBaseSensorId(const SensorId& base_sensor_id) {
    base_sensor_id_ = base_sensor_id;
  }
  const aslam::SensorId& getBaseSensorId() const {
    return base_sensor_id_;
  }

  void set_T_B_S(const Transformation& T_B_S, const SensorId& base_sensor_id) {
    T_B_S_ = T_B_S;
    base_sensor_id_ = base_sensor_id;
  }

  // Virtual
  virtual uint8_t getSensorType() const = 0;
  virtual std::string getSensorTypeString() const = 0;

  bool isValid() const;

  bool deserialize(const YAML::Node& sensor_node) override;
  void serialize(YAML::Node* sensor_node_ptr) const override;

 private:
  virtual bool isValidImpl() const = 0;
  virtual bool isEqualImpl(const Sensor& other, const bool verbose) const = 0;

  virtual bool loadFromYamlNodeImpl(const YAML::Node& sensor_node) = 0;
  virtual void saveToYamlNodeImpl(YAML::Node* sensor_node) const = 0;

 protected:
  // The id of this sensor.
  SensorId id_;
  std::string topic_;
  std::string description_;
  SensorId base_sensor_id_;
  Transformation T_B_S_;
};

}  // namespace aslam

#endif  // ASLAM_COMMON_SENSOR_H_
