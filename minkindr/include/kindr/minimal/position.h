#ifndef KINDR_MINIMAL_POSITION_H
#define KINDR_MINIMAL_POSITION_H

#include <Eigen/Dense>

namespace kindr {
namespace minimal {

template <typename Scalar>
using PositionTemplate = Eigen::Matrix<Scalar, 3, 1>;

typedef PositionTemplate<double> Position;

} // namespace minimal
} // namespace kindr

#endif /* KINDR_MINIMAL_POSITION_H */
