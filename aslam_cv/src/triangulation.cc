#include "aslam/triangulation/triangulation.h"

namespace aslam {

TriangulationResult::Status TriangulationResult::SUCCESSFUL =
    TriangulationResult::Status::SUCCESSFUL;
TriangulationResult::Status TriangulationResult::TOO_FEW_MEASUREMENTS =
    TriangulationResult::Status::TOO_FEW_MEASUREMENTS;
TriangulationResult::Status TriangulationResult::UNOBSERVABLE =
    TriangulationResult::Status::UNOBSERVABLE;
TriangulationResult::Status TriangulationResult::UNINITIALIZED =
    TriangulationResult::Status::UNINITIALIZED;
}  // namespace aslam

