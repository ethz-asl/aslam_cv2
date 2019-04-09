#include <aslam/common/sensor.h>

namespace aslam {
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

  // TODO(smauq): Fix
  /*std::string sensor_type_as_string;
  if (!YAML::safeGet(
          sensor_node, static_cast<std::string>(kYamlFieldNameSensorType),
          &sensor_type_as_string)) {
    LOG(ERROR) << "Unable to retrieve the sensor type from the given "
               << "YAML node.";
    return false;
  }
  sensor_type_ = stringToSensorType(sensor_type_as_string);*/

  if (!YAML::safeGet(
          sensor_node, static_cast<std::string>(kYamlFieldNameHardwareId),
          &topic_)) {
    LOG(ERROR) << "Unable to retrieve the sensor topic from the given "
               << "YAML node.";
    return false;
  }
  CHECK(!topic_.empty()) << "A sensor needs a non-empty topic.";

  return loadFromYamlNodeImpl(sensor_node);
}

void Sensor::serialize(YAML::Node* sensor_node_ptr) const {
  YAML::Node& sensor_node = *CHECK_NOTNULL(sensor_node_ptr);

  CHECK(id_.isValid());
  sensor_node[static_cast<std::string>(kYamlFieldNameId)] = id_.hexString();
  // TODO(smauq): Fix
  /*sensor_node[static_cast<std::string>(kYamlFieldNameSensorType)] =
      sensorTypeToString(sensor_type_);*/
  CHECK(!topic_.empty());
  sensor_node[static_cast<std::string>(kYamlFieldNameHardwareId)] = topic_;

  saveToYamlNodeImpl(&sensor_node);
}
}
