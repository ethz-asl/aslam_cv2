#include <aslam/common/sensor.h>

namespace aslam {
Sensor::Sensor(int sensor_type)
    : sensor_type_(sensor_type) {}

Sensor::Sensor(const SensorId& id)
    : id_(id) {
  CHECK(id_.isValid());
};

Sensor::Sensor(const SensorId& id, int sensor_type)
    : id_(id), sensor_type_(sensor_type) {
  CHECK(id_.isValid());
};

Sensor::Sensor(const SensorId& id, int sensor_type, const std::string& topic)
    : id_(id), sensor_type_(sensor_type), topic_(topic) {
  CHECK(id_.isValid());
};

void Sensor::setId(const SensorId& id) {
  id_ = id;
  CHECK(id_.isValid());
};

bool Sensor::isValid() const {
  if (!id_.isValid()) {
    LOG(ERROR) << "Invalid sensor id.";
    return false;
  }
  return isValidImpl();
}

void Sensor::setRandom() {
  generateId(&id_);
  setRandomImpl();
}

bool Sensor::deserialize(const YAML::Node& sensor_node) {
  CHECK(!sensor_node.IsNull());
  std::string id_as_string;
  if (YAML::safeGet(
        sensor_node, static_cast<std::string>(kYamlFieldNameId),
        &id_as_string)) {
    CHECK(!id_as_string.empty());
    CHECK(id_.fromHexString(id_as_string));
  } else {
    LOG(WARNING) << "Unable to find an ID field. Generating a new random id.";
    generateId(&id_);
  }
  CHECK(id_.isValid());

  std::string sensor_type_as_string;
  if (YAML::safeGet(
        sensor_node, static_cast<std::string>(kYamlFieldNameSensorType),
        &sensor_type_as_string)) {
    try {
      sensor_type_ = std::stoi(sensor_type_as_string);
    } catch(const std::exception& e) {
      LOG(ERROR)
          << "Exception " << e.what() << ", sensor type "
          << sensor_type_as_string << " must be an integer.";
      return false;
    }
  } else {
    LOG(WARNING)
        << "Unable to retrieve the sensor type, setting to unknown.";
    sensor_type_ = SensorType::kUnknown;
  }

  if (sensor_type_ != aslam::SensorType::kNCamera && !YAML::safeGet(
        sensor_node, static_cast<std::string>(kYamlFieldNameTopic), &topic_)) {
    LOG(WARNING)
        << "Unable to retrieve the sensor topic.";
  }

  return loadFromYamlNodeImpl(sensor_node);
}

void Sensor::serialize(YAML::Node* sensor_node_ptr) const {
  YAML::Node& sensor_node = *CHECK_NOTNULL(sensor_node_ptr);

  CHECK(id_.isValid());
  sensor_node[static_cast<std::string>(kYamlFieldNameId)] = id_.hexString();
  sensor_node[static_cast<std::string>(kYamlFieldNameSensorType)] =
      std::to_string(sensor_type_);
  if (sensor_type_ != aslam::SensorType::kNCamera) {
    sensor_node[static_cast<std::string>(kYamlFieldNameTopic)] = topic_;
  }

  saveToYamlNodeImpl(&sensor_node);
}
}  // namespace aslam
