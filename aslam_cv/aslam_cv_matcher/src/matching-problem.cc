#include <gflags/gflags.h>

DEFINE_bool(matcher_store_all_tested_pairs, false, "If true, every tested match pair, regardless"
    " of whether it fulfilled the matching criteria, is stored in a list as a member of the "
    " matching problem and can be retrieved after the matching for debugging and/or visualization "
    " purposes.");
