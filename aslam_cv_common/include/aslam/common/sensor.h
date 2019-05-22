#ifndef ASLAM_COMMON_SENSOR_H_
#define ASLAM_COMMON_SENSOR_H_

#include <string>

#include <aslam/common/macros.h>
#include <aslam/common/unique-id.h>
#include <aslam/common/yaml-file-serialization.h>
#include <aslam/common/yaml-serialization.h>

namespace aslam {

enum SensorType : uint8_t { kUnknown, kNCamera, kCamera };

constexpr char kNCameraIdentifier[] = "NCAMERA";
constexpr char kCameraIdentifier[] = "CAMERA";

constexpr char kYamlFieldNameId[] = "id";
constexpr char kYamlFieldNameSensorType[] = "sensor_type";
constexpr char kYamlFieldNameTopic[] = "topic";
constexpr char kYamlFieldNameDescription[] = "description";

class Sensor : public YamlFileSerializable {
 public:
  ASLAM_POINTER_TYPEDEFS(Sensor);

  Sensor(){};
  explicit Sensor(const SensorId& id);
  explicit Sensor(const SensorId& id, const std::string& topic);

  virtual ~Sensor() = default;

  Sensor(const Sensor& other)
      : id_(other.id_),
        topic_(other.topic_),
        description_(other.description_) {}
  void operator=(const Sensor& other) {
    id_ = other.id_;
    topic_ = other.topic_;
    description_ = other.description_;
  }

  bool operator==(const Sensor& other) const;
  bool operator!=(const Sensor& other) const;
  bool isEqual(const Sensor& other) const;

  virtual Sensor::Ptr cloneAsSensor() const = 0;

  // Set and get the sensor id.
  void setId(const SensorId& id);
  const SensorId& getId() const {
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
  void setDescription(const std::string& description);
  const std::string& getDescription() const {
    return description_;
  }

  // Virtual
  virtual uint8_t getSensorType() const = 0;
  virtual std::string getSensorTypeString() const = 0;

  bool isValid() const;

  bool deserialize(const YAML::Node& sensor_node) override;
  void serialize(YAML::Node* sensor_node_ptr) const override;

  void setRandom();

 private:
  virtual bool isValidImpl() const = 0;
  virtual void setRandomImpl() = 0;
  virtual bool isEqualImpl(const Sensor& other) const = 0;

  virtual bool loadFromYamlNodeImpl(const YAML::Node& sensor_node) = 0;
  virtual void saveToYamlNodeImpl(YAML::Node* sensor_node) const = 0;

 protected:
  // The id of this sensor.
  SensorId id_;
  std::string topic_;
  std::string description_;
};

}  // namespace aslam

#endif  // ASLAM_COMMON_SENSOR_H_
