#include <aslam/common/sensor.h>

namespace aslam {

Sensor::Sensor(const SensorId& id) : id_(id) {
  CHECK(id_.isValid());
};

Sensor::Sensor(const SensorId& id, const std::string& topic)
    : id_(id), topic_(topic) {
  CHECK(id_.isValid());
};

void Sensor::setId(const SensorId& id) {
  id_ = id;
  CHECK(id_.isValid());
};

void Sensor::setDescription(const std::string& description) {
  description_ = description;
}

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
  if (!sensor_node.IsDefined() || sensor_node.IsNull()) {
    LOG(ERROR) << "Invalid YAML node for sensor deserialization.";
    return false;
  }

  if (!sensor_node.IsMap()) {
    LOG(WARNING) << "Sensor YAML node must be a map.";
    return false;
  }

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

  if (!YAML::safeGet(
          sensor_node, static_cast<std::string>(kYamlFieldNameTopic),
          &topic_)) {
    LOG(WARNING) << "Unable to retrieve the sensor topic for sensor " << id_;
  }

  if (!YAML::safeGet(
          sensor_node, static_cast<std::string>(kYamlFieldNameDescription),
          &description_)) {
    LOG(WARNING) << "Unable to retrieve the sensor description.";
  }

  return loadFromYamlNodeImpl(sensor_node);
}

void Sensor::serialize(YAML::Node* sensor_node_ptr) const {
  YAML::Node& sensor_node = *CHECK_NOTNULL(sensor_node_ptr);

  CHECK(id_.isValid());
  sensor_node[static_cast<std::string>(kYamlFieldNameId)] = id_.hexString();
  sensor_node[static_cast<std::string>(kYamlFieldNameSensorType)] =
      getSensorTypeString();
  sensor_node[static_cast<std::string>(kYamlFieldNameTopic)] = topic_;
  sensor_node[static_cast<std::string>(kYamlFieldNameDescription)] =
      description_;

  saveToYamlNodeImpl(&sensor_node);
}

bool Sensor::operator==(const Sensor& other) const {
  return isEqual(other);
}

bool Sensor::operator!=(const Sensor& other) const {
  return !isEqual(other);
}

bool Sensor::isEqual(const Sensor& other) const {
  bool is_equal = true;
  is_equal &= id_ == other.id_;
  is_equal &= topic_ == other.topic_;
  is_equal &= description_ == other.description_;
  is_equal &= getSensorType() == other.getSensorType();
  is_equal &= isEqualImpl(other);
  return is_equal;
}

}  // namespace aslam
