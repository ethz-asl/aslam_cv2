#include <aslam/common/sensor.h>

namespace aslam {

Sensor::Sensor() : topic_(""), description_("") {
  generateId(&id_);
  base_sensor_id_.setInvalid();
  T_B_S_.setIdentity();
}

Sensor::Sensor(const SensorId& id) : id_(id), topic_(""), description_("") {
  CHECK(id.isValid());
  base_sensor_id_.setInvalid();
  T_B_S_.setIdentity();
}

Sensor::Sensor(const SensorId& id, const std::string& topic)
    : id_(id), topic_(topic), description_("") {
  CHECK(id.isValid());
  base_sensor_id_.setInvalid();
  T_B_S_.setIdentity();
}

Sensor::Sensor(
    const SensorId& id, const std::string& topic,
    const std::string& description)
    : id_(id), topic_(topic), description_(description) {
  CHECK(id.isValid());
  base_sensor_id_.setInvalid();
  T_B_S_.setIdentity();
}

bool Sensor::isValid() const {
  if (!id_.isValid()) {
    LOG(ERROR) << "Invalid sensor id.";
    return false;
  }
  if (getSensorType() != SensorType::kNCamera) {
    if (!base_sensor_id_.isValid()) {
      LOG(ERROR) << "Invalid base sensor id.";
      return false;
    }
  }

  return isValidImpl();
}

bool Sensor::deserialize(const YAML::Node& sensor_node) {
  if (!sensor_node.IsDefined() || sensor_node.IsNull()) {
    LOG(ERROR) << "Invalid YAML node for sensor deserialization.";
    return false;
  }

  if (!sensor_node.IsMap()) {
    LOG(ERROR) << "Sensor YAML node must be a map.";
    return false;
  }

  std::string id_as_string;
  if (YAML::safeGet(
          sensor_node, static_cast<std::string>(kYamlFieldNameId),
          &id_as_string)) {
    if (id_as_string.empty()) {
      LOG(ERROR) << "Found empty sensor id field, please provide a sensor id "
                 << "or remove the field to automatically generate an id.";
      return false;
    }

    if (!id_.fromHexString(id_as_string)) {
      LOG(ERROR) << "Failed to deserialize sensor id " << id_as_string << ".";
      return false;
    }
  } else {
    LOG(WARNING)
        << "Unable to find an ID field. Generating a new random id. With "
        << "single session mapping this will have little consequences. "
        << "However, for multisession mapping this will disassociate the same "
        << "sensor appearing across sessions.";
    generateId(&id_);
  }
  CHECK(id_.isValid());

  if (getSensorType() == SensorType::kNCamera) {
    topic_.clear();
    LOG(WARNING) << "Ignoring NCamera topic, as it's a meta sensor.";
  } else if (!YAML::safeGet(
                 sensor_node, static_cast<std::string>(kYamlFieldNameTopic),
                 &topic_)) {
    if (getSensorType() != SensorType::kNCamera) {
      LOG(WARNING) << "Unable to retrieve the sensor topic for sensor " << id_;
    }
  }

  if (!YAML::safeGet(
          sensor_node, static_cast<std::string>(kYamlFieldNameDescription),
          &description_)) {
    LOG(WARNING) << "Unable to retrieve sensor description for " << id_ << ".";
  }

  if (sensor_node[static_cast<std::string>(kYamlFieldNameBaseSensorId)]) {
    id_as_string.clear();
    CHECK(YAML::safeGet(
        sensor_node, static_cast<std::string>(kYamlFieldNameBaseSensorId),
        &id_as_string));
    if (getSensorType() != SensorType::kNCamera) {
      if (id_as_string.empty()) {
        LOG(ERROR) << "Found empty base sensor id field.";
        return false;
      }

      if (!base_sensor_id_.fromHexString(id_as_string)) {
        LOG(ERROR) << "Failed to deserialize base sensor id " << id_as_string
                   << " of sensor " << id_ << ".";
        return false;
      }
    } else {
      base_sensor_id_.setInvalid();
      LOG(WARNING)
          << "Ignoring NCamera base sensor id, as it's a meta sensor. Instead "
          << "set a base sensor and transformation for the individual cameras.";
    }
  } else if (getSensorType() != SensorType::kNCamera) {
    LOG(ERROR) << "No base sensor id provided for sensor " << id_ << ". If "
               << "this sensor is a base sensor, set the base_sensor_id field "
               << "to the same value as the id.";
    return false;
  }

  if (id_ != base_sensor_id_) {
    Eigen::Matrix4d input_matrix;
    if (sensor_node["T_B_S"]) {
      if (getSensorType() == SensorType::kNCamera) {
        LOG(ERROR)
            << "At the moment NCameras are only a meta sensor type and "
            << "the calibration is not applied to the individual cameras. "
            << "Only the T_B_S or T_S_B in each camera is used.";
        return false;
      }

      CHECK(YAML::safeGet(sensor_node, "T_B_S", &input_matrix));
      T_B_S_ = aslam::Transformation(input_matrix);
    } else if (sensor_node["T_S_B"]) {
      if (getSensorType() == SensorType::kNCamera) {
        LOG(ERROR)
            << "At the moment NCameras are only a meta sensor type and "
            << "the calibration is not applied to the individual cameras. "
            << "Only the T_B_S or T_S_B in each camera is used.";
        return false;
      }

      CHECK(YAML::safeGet(sensor_node, "T_S_B", &input_matrix));
      T_B_S_ = aslam::Transformation(input_matrix).inverse();
    } else {
      T_B_S_.setIdentity();
      if (getSensorType() != SensorType::kNCamera) {
        LOG(WARNING) << "Unable to get extrinsic transformation T_B_S or T_S_B "
                     << "for sensor " << id_ << ". Assuming identity!";
      }
    }
  } else {
    if (sensor_node["T_B_S"] || sensor_node["T_S_B"]) {
      LOG(ERROR) << "Base sensor can't have a transformation relative to "
                 << "another sensor. It's always assumed to be unity.";
      return false;
    }

    T_B_S_.setIdentity();
  }

  return loadFromYamlNodeImpl(sensor_node);
}

void Sensor::serialize(YAML::Node* sensor_node_ptr) const {
  YAML::Node& sensor_node = *CHECK_NOTNULL(sensor_node_ptr);

  CHECK(id_.isValid());
  sensor_node[static_cast<std::string>(kYamlFieldNameId)] = id_.hexString();
  sensor_node[static_cast<std::string>(kYamlFieldNameSensorType)] =
      getSensorTypeString();
  sensor_node[static_cast<std::string>(kYamlFieldNameDescription)] =
      description_;

  if (getSensorType() != SensorType::kNCamera) {
    sensor_node[static_cast<std::string>(kYamlFieldNameTopic)] = topic_;
    sensor_node[static_cast<std::string>(kYamlFieldNameBaseSensorId)] =
        base_sensor_id_.hexString();
    if (id_ != base_sensor_id_) {
      sensor_node["T_B_S"] = T_B_S_.getTransformationMatrix();
    }
  }

  saveToYamlNodeImpl(&sensor_node);
}

bool Sensor::operator==(const Sensor& other) const {
  return isEqual(other, true /*verbose*/);
}

bool Sensor::operator!=(const Sensor& other) const {
  return !isEqual(other, true /*verbose*/);
}

bool Sensor::isEqual(const Sensor& other, const bool verbose) const {
  bool is_equal = true;
  is_equal &= id_ == other.id_;
  is_equal &= topic_ == other.topic_;
  is_equal &= description_ == other.description_;
  is_equal &= getSensorType() == other.getSensorType();
  is_equal &= base_sensor_id_ == other.base_sensor_id_;

  const Eigen::Matrix4d T_B_S_diff =
      T_B_S_.getTransformationMatrix() - other.T_B_S_.getTransformationMatrix();
  is_equal &= T_B_S_diff.cwiseAbs().maxCoeff() < common::macros::kEpsilon;

  if (!is_equal) {
    LOG_IF(WARNING, verbose) << "this sensor: "
                             << "\n id: " << id_ << "\n topic: " << topic_
                             << "\n description: " << description_
                             << "\n sensor_type: " << getSensorTypeString();
    LOG_IF(WARNING, verbose)
        << "other sensor: "
        << "\n id: " << id_ << "\n topic: " << other.topic_
        << "\n description: " << other.description_
        << "\n sensor_type: " << other.getSensorTypeString();
  }

  // optimize to avoid unncessary comparisons
  if (is_equal) {
    is_equal &= isEqualImpl(other, verbose);
  }

  return is_equal;
}

}  // namespace aslam
