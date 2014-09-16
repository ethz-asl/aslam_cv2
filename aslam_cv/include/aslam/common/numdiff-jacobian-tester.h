#ifndef ASLAM_CV_NUMDIFF_JACOBIAN_TESTER_H_
#define ASLAM_CV_NUMDIFF_JACOBIAN_TESTER_H_
#include <Eigen/Dense>
#include <aslam/common/eigen-helpers.h>


#define NUMDIFF_DEBUG_OUTPUT false

namespace aslam {
namespace common {

/// \brief Test macro and numerical differentiator to unit test Jacobian implementations.
/// Example:
///
///   struct Functor : public aslam::common::NumDiffFunctor<2, 3> {
///    Functor(MyClass::Ptr my_class) : my_class_(my_class) {};
///    virtual ~Functor() {};
///
///    void functional(const typename NumDiffFunctor::InputType& x,
///                    typename NumDiffFunctor::ValueType& fvec,
///                    typename NumDiffFunctor::JacobianType* Jout) const {
///        fvec = getValue(x,my_class->params);
///        Jout = my_class_->getAnalyticalJacobian(x, my_class->params);
///    };
///
///    MyClass::Ptr my_class_;
///  };
///
///  double stepsize = 1e-3;
///  double test_tolerance = 1e-2;
///  Eigen::Vector2d x0(0.0, 1.0); // Evaluation point
///  TEST_JACOBIAN_FINITE_DIFFERENCE(Functor, x0, stepsize, tolerance, my_class_);
#define TEST_JACOBIAN_FINITE_DIFFERENCE(FUNCTOR_TYPE, X, STEP, TOLERANCE, ...) \
    do {\
      FUNCTOR_TYPE functor(__VA_ARGS__); \
      aslam::common::NumericalDiff<FUNCTOR_TYPE> numDiff(functor, STEP); \
      typename FUNCTOR_TYPE::JacobianType Jnumeric;\
      numDiff.getJacobianNumerical(X, Jnumeric); \
      typename FUNCTOR_TYPE::JacobianType Jsymbolic; \
      functor.getJacobian(X, &Jsymbolic); \
      EXPECT_NEAR_EIGEN(Jnumeric, Jsymbolic, TOLERANCE); \
      if (NUMDIFF_DEBUG_OUTPUT) std::cout << "Jnumeric: " << Jnumeric << "\n"; \
      if (NUMDIFF_DEBUG_OUTPUT) std::cout << "Jsymbolic: " << Jsymbolic << "\n"; \
    } while (0)


// Functor base for numerical differentiation.
template<int NY, int NX, typename _Scalar = double>
struct NumDiffFunctor {
  // Type definitions.
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };

  typedef _Scalar Scalar;
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  NumDiffFunctor() {};
  virtual ~NumDiffFunctor() {};

  virtual void functional(const typename NumDiffFunctor::InputType& x,
                          typename NumDiffFunctor::ValueType& fvec,
                          typename NumDiffFunctor::JacobianType* Jout) const = 0;

  void operator()(const InputType& x, ValueType& fvec) const {
    functional(x, fvec, nullptr);
  };

  void getJacobian(const InputType& x,
                   JacobianType* out_jacobian) const {
    ValueType fvec;
    functional(x, fvec, out_jacobian);
  };
};

/// Differentiation schemes.
enum NumericalDiffMode {
  Forward,
  Central,
  CentralSecond
};

/// Modified numerical differentiation code from unsupported/Eigen library
template<typename _Functor, NumericalDiffMode mode = CentralSecond>
class NumericalDiff : public _Functor {

 public:
  typedef _Functor Functor;
  typedef typename Functor::Scalar Scalar;
  typedef typename Functor::InputType InputType;
  typedef typename Functor::ValueType ValueType;
  typedef typename Functor::JacobianType JacobianType;

  NumericalDiff(const Functor& f, Scalar _epsfcn = 0.)
      : Functor(f),
        epsfcn(_epsfcn) {}

  enum {
    InputsAtCompileTime = Functor::InputsAtCompileTime,
    ValuesAtCompileTime = Functor::ValuesAtCompileTime
  };

  void getJacobianNumerical(const InputType& _x, JacobianType &jac) const {
    using std::sqrt;
    using std::abs;
    /* Local variables */
    Scalar h;

    const typename InputType::Index n = _x.size();
    const Scalar eps = sqrt(((std::max)(epsfcn, Eigen::NumTraits<Scalar>::epsilon())));

    // Build jacobian.
    InputType x = _x;
    ValueType val1, val2, val3, val4;

    for (int j = 0; j < n; ++j) {
      h = eps * abs(x[j]);
      if (h == 0.) {
        h = eps;
      }

      switch (mode) {
        case Forward:
                        Functor::operator()(x, val1);
          x[j] += h;    Functor::operator()(x, val2);
          x[j] = _x[j];
          jac.col(j) = (val2 - val1) / h;
          break;
        case Central:
          x[j] += h;     Functor::operator()(x, val2);
          x[j] -= 2 * h; Functor::operator()(x, val1);
          x[j] = _x[j];
          jac.col(j) = (val2 - val1) / (2 * h);
          break;
        case CentralSecond:
          x[j] += 2.0 * h; Functor::operator()(x, val1);
          x[j] -= h;       Functor::operator()(x, val2);
          x[j] -= 2.0 * h; Functor::operator()(x, val3);
          x[j] -= h;       Functor::operator()(x, val4);
          x[j] = _x[j];

          jac.col(j) = ( (8.0*val2) + val4 - val1 - (8.0*val3))/(h * 12.0);
          break;
        default:
          eigen_assert(false);
      };
    }
  }
 private:
  Scalar epsfcn;

  NumericalDiff& operator=(const NumericalDiff&);
};


}  // namespace common
}  // namespace aslam

#endif  // ASLAM_CV_NUMDIFF_JACOBIAN_TESTER_H_
