#include "aslam/triangulation/triangulation.h"

namespace aslam {

TriangulationResult::Status TriangulationResult::SUCCESSFUL =
    TriangulationResult::Status::kSuccessful;
TriangulationResult::Status TriangulationResult::TOO_FEW_MEASUREMENTS =
    TriangulationResult::Status::kTooFewMeasurments;
TriangulationResult::Status TriangulationResult::UNOBSERVABLE =
    TriangulationResult::Status::kUnobservable;
TriangulationResult::Status TriangulationResult::UNINITIALIZED =
    TriangulationResult::Status::kUninitialized;
}  // namespace aslam

