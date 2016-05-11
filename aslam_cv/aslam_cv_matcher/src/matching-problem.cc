#include "aslam/matcher/matching-problem.h"

#include <gflags/gflags.h>

DEFINE_bool(matcher_store_all_tested_pairs, false, "If true, every tested match pair, regardless"
    " of whether it fulfilled the matching criteria, is stored in a list and can be retrieved "
    "after the matching for debugging and/or visualization purposes.");

namespace aslam {

MatchingProblem::MatchingProblem() : store_tested_pairs_(FLAGS_matcher_store_all_tested_pairs) {}

}  // namespace aslam
