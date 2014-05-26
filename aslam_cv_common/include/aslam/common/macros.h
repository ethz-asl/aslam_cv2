#ifndef ASLAM_COMMON_MACROS_H_
#define ASLAM_COMMON_MACROS_H_

#include <memory>

#define ASLAM_DISALLOW_EVIL_CONSTRUCTORS(TypeName)     \
  TypeName(const TypeName&);                           \
  void operator=(const TypeName&)

#define ASLAM_POINTER_TYPEDEFS(TypeName)               \
  typedef std::shared_ptr<TypeName> Ptr;               \
  typedef std::shared_ptr<const TypeName> ConstPtr     \

#endif ASLAM_COMMON_MACROS_H_
