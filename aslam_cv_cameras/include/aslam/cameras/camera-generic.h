#ifndef ASLAM_CAMERAS_GENERIC_CAMERA_H_
#define ASLAM_CAMERAS_GENERIC_CAMERA_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/crtp-clone.h>
#include <aslam/common/macros.h>
#include <aslam/common/types.h>

namespace aslam {

// Forward declarations.
class MappedUndistorter;
class NCamera;

/// \class GenericCamera
/// \brief An implementation of the generic camera model.
///
///
///  Reference: 
class GenericCamera : public aslam::Cloneable<Camera, GenericCamera> {
  friend class NCamera;

 public:
  ASLAM_POINTER_TYPEDEFS(GenericCamera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum Parameters { 
    kCalibrationMinX = 0,
    kCalibrationMinY = 1,
    kCalibrationMaxX = 2,
    kCalibrationMaxY = 3,
    kGridWidth = 4,
    kGridHeight = 5,
    kGrid = 6,
  };

  // TODO(slynen) Enable commented out PropertyTree support
  // GenericCamera(const sm::PropertyTree& config);

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

 public:
  /// \brief Empty constructor for serialization interface.
  GenericCamera();

  /// Copy constructor for clone operation.
  GenericCamera(const GenericCamera& other) = default;
  void operator=(const GenericCamera&) = delete;

 public:
  /// \brief Construct a GenericCamera while supplying distortion. Distortion is removed.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  /// @param[in] distortion   Pointer to the distortion model.
  GenericCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height,
                aslam::Distortion::UniquePtr& distortion);
                
  /// \brief Construct a GenericCamera without distortion.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  GenericCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height);

  virtual ~GenericCamera() {};

  /// \brief Convenience function to print the state using streams.
  friend std::ostream& operator<<(std::ostream& out, const GenericCamera& camera);

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project euclidean points
  /// @{

  /// \brief Compute the 3d bearing vector in euclidean coordinates given a keypoint in
  ///        image coordinates. Uses the projection model.
  ///        The result might be in normalized image plane for some camera implementations but not
  ///        for the general case.
  /// @param[in]  keypoint     Keypoint in image coordinates.
  /// @param[out] out_point_3d Bearing vector in euclidean coordinates
  /// @return Contains if back-projection was possible
  virtual bool backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                            Eigen::Vector3d* out_point_3d) const;

  bool backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const; 

  /// \brief Checks the success of a projection operation and returns the result in a
  ///        ProjectionResult object.
  /// @param[in] keypoint Keypoint in image coordinates.
  /// @param[in] point_3d Projected point in euclidean.
  /// @return The ProjectionResult object contains details about the success of the projection.
  template <typename DerivedKeyPoint, typename DerivedPoint3d>
  inline const ProjectionResult evaluateProjectionResult(
      const Eigen::MatrixBase<DerivedKeyPoint>& keypoint,
      const Eigen::MatrixBase<DerivedPoint3d>& point_3d) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Functional methods to project and back-project points
  /// @{

  // Get the overloaded non-virtual project3Functional(..) from base into scope.
  using Camera::project3Functional;

  /// \brief Template version of project3Functional.
  template <typename ScalarType, typename DistortionType,
            typename MIntrinsics, typename MDistortion>
  const ProjectionResult project3Functional(
      const Eigen::Matrix<ScalarType, 3, 1>& point_3d,
      const Eigen::MatrixBase<MIntrinsics>& intrinsics_external,
      const Eigen::MatrixBase<MDistortion>& distortion_coefficients_external,
      Eigen::Matrix<ScalarType, 2, 1>* out_keypoint) const;

  /// \brief This function projects a point into the image using the intrinsic parameters
  ///        that are passed in as arguments. If any of the Jacobians are nonnull, they
  ///        should be filled in with the Jacobian with respect to small changes in the argument.
  /// @param[in]  point_3d                The point in euclidean coordinates.
  /// @param[in]  intrinsics_external     External intrinsic parameter vector.
  ///                                     NOTE: If nullptr, use internal distortion parameters.
  /// @param[in]  distortion_coefficients_external External distortion parameter vector.
  ///                                     Parameter is ignored is no distortion is active.
  ///                                     NOTE: If nullptr, use internal distortion parameters.
  /// @param[out] out_keypoint            The keypoint in image coordinates.
  /// @param[out] out_jacobian_point      The Jacobian wrt. to changes in the euclidean point.
  ///                                       nullptr: calculation is skipped.
  /// @param[out] out_jacobian_intrinsics The Jacobian wrt. to changes in the intrinsics.
  ///                                       nullptr: calculation is skipped.
  /// @param[out] out_jacobian_distortion The Jacobian wrt. to changes in the distortion parameters.
  ///                                       nullptr: calculation is skipped.
  /// @return Contains information about the success of the projection. Check
  ///         \ref ProjectionResult for more information.
  virtual const ProjectionResult project3Functional(
      const Eigen::Ref<const Eigen::Vector3d>& point_3d,
      const Eigen::VectorXd* intrinsics_external,
      const Eigen::VectorXd* distortion_coefficients_external,
      Eigen::Vector2d* out_keypoint,
      Eigen::Matrix<double, 2, 3>* out_jacobian_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to support unit testing.
  /// @{

  /// \brief Creates a random valid keypoint..
  virtual Eigen::Vector2d createRandomKeypoint() const;

  /// \brief Creates a random visible point. Negative depth means random between
  ///        0 and 100 meters.
  virtual Eigen::Vector3d createRandomVisiblePoint(double depth) const;

  /// @}

 public:
  //////////////////////////////////////////////////////////////
  /// \name Methods to access intrinsics.
  /// @{

  /// \brief The horizontal focal length in pixels.
  double calibrationMinX() const { return intrinsics_[Parameters::kCalibrationMinX]; };
  /// \brief The vertical focal length in pixels.
  double calibrationMinY() const { return intrinsics_[Parameters::kCalibrationMinY]; };
  /// \brief The horizontal image center in pixels.
  double calibrationMaxX() const { return intrinsics_[Parameters::kCalibrationMaxX]; };
  /// \brief The vertical image center in pixels.
  double calibrationMaxY() const { return intrinsics_[Parameters::kCalibrationMaxY]; };
  /// \brief The horizontal image center in pixels.
  double gridWidth() const { return intrinsics_[Parameters::kGridWidth]; };
  /// \brief The vertical image center in pixels.
  double gridHeight() const { return intrinsics_[Parameters::kGridHeight]; };
  /// \brief The total size of the grid.
  double gridSize() const { return gridWidth() * gridHeight(); };
  /// \brief The centerpoint of the calibrated area.
  Eigen::Vector2d centerOfCalibratedArea() const {
    return Eigen::Vector2d(
      0.5 * (calibrationMinX() + calibrationMaxX() + 1.),
      0.5 * (calibrationMinY() + calibrationMaxY() + 1.)
    );
  };

  /// \brief Returns the number of intrinsic parameters this camera model over which can be optimized.
  /// Since in the generic model the intrinsics are fixed, this is 0.
  inline static constexpr int parameterCount() {
      return 0;
  }

  /// \brief Returns the number of intrinsic parameters used in this camera model.
  inline virtual int getParameterSize() const {
      return intrinsics_.size();
  }

  /// Static function that checks whether the given intrinsic parameters are valid for this model.
  static bool areParametersValid(const Eigen::VectorXd& parameters);

  /// Function to check whether the given intrinsic parameters are valid for
  /// this model.
  virtual bool intrinsicsValid(const Eigen::VectorXd& intrinsics) const;

  /// Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras
  virtual void printParameters(std::ostream& out, const std::string& text) const;

  /// @}

  /// \brief Create a test camera object for unit testing.
  template <typename DistortionType>
  static GenericCamera::Ptr createTestCamera() {
    return GenericCamera::Ptr(
        std::move(createTestCameraUnique<DistortionType>()));
  }

  /// \brief Create a test camera object for unit testing.
  template <typename DistortionType>
  static GenericCamera::UniquePtr createTestCameraUnique() {
    YAML::Node node = YAML::Load("{type: generic, intrinsics: {rows: 6, cols: 1, data: [15,15,736,464,16,11]}, image_height: 480, image_width: 752,  grid: {rows: 176, cols: 3, data: [-0.73680864697228, -0.48562625042812, 0.47040425448969, -0.65643352034764, -0.49748019131363, 0.56710536288644, -0.57529862830316, -0.51573877806511, 0.63485825274026, -0.4829821466507, -0.53015078136865, 0.69689912830401, -0.38444374137903, -0.54243549487623, 0.74697171540352, -0.28086806363103, -0.55134648666921, 0.78557633777353, -0.1721103096518, -0.55960036817312, 0.810694436426, -0.061722237678026, -0.56287084396511, 0.82423708870083, 0.04989935551493, -0.56445602355111, 0.82395354953788, 0.16062533537446, -0.56247278237593, 0.81106341966712, 0.26901089536198, -0.55820951520686, 0.78487914694497, 0.37276222195738, -0.55095630336118, 0.74665619776975, 0.47204414033422, -0.54025171159243, 0.6966365032767, 0.56275855431752, -0.52822827676352, 0.6358283551163, 0.64784236585731, -0.51131660361745, 0.5646730026002, 0.71758270837233, -0.50356690402212, 0.48113971964348, -0.74482660879238, -0.3909445771263, 0.54073622076792, -0.67189174532953, -0.40740587055911, 0.61853208420583, -0.58631871071313, -0.42173191952584, 0.69164482035272, -0.49253828501389, -0.43359083149535, 0.75458931124072, -0.39178761654292, -0.44396537591327, 0.80585185270855, -0.28515926095472, -0.45196677267159, 0.84522791736465, -0.17454731383209, -0.45740159278634, 0.87196159213036, -0.061229766592886, -0.46096098857752, 0.8853055307026, 0.052817721847604, -0.46193577048946, 0.88533927519406, 0.16604198640003, -0.46064826643592, 0.87191354696547, 0.27681523432077, -0.45693512124914, 0.84533048035486, 0.38346886842054, -0.45093315196242, 0.80598444117334, 0.48430956889612, -0.44281560227242, 0.75455853574111, 0.57829427650963, -0.43228773251534, 0.69188369403603, 0.66344720274583, -0.42005923335663, 0.61918337319454, 0.74000379826004, -0.40493021217735, 0.53705297860334, -0.75990179436045, -0.29680790462239, 0.57832026653182, -0.68243718745857, -0.31039318190435, 0.66176699660902, -0.5953610787439, -0.32079809410701, 0.73663679566949, -0.49982352918459, -0.33047927303425, 0.80059970632533, -0.39729683367312, -0.33831840695849, 0.8530509254824, -0.28865961220064, -0.3445488781225, 0.89328701931054, -0.17603626334314, -0.34914465656895, 0.92038537731623, -0.060806842321716, -0.35179614609808, 0.93409956616915, 0.055267105910384, -0.35278842202231, 0.93406952433494, 0.17053574142323, -0.35186586124566, 0.92038468945712, 0.28323547455551, -0.34918832863142, 0.89322179614075, 0.39168807991485, -0.34473134068351, 0.85307722440772, 0.49450477832578, -0.33842753159012, 0.80058218196184, 0.58997181479829, -0.33060847079331, 0.73663511780481, 0.67706344707325, -0.32104402890598, 0.66220527039665, 0.75420409067053, -0.31050005146438, 0.57858958481505, -0.76812264873815, -0.2009072021158, 0.60796701607364, -0.68992177844774, -0.20759326624913, 0.69347889328481, -0.60163662553573, -0.21557203106791, 0.76913072376238, -0.50503800010051, -0.22209778457673, 0.83403188940267, -0.40098308456185, -0.22779772730831, 0.88731097216727, -0.29110727643508, -0.23216340458488, 0.92809304877157, -0.17695562751952, -0.23540952216918, 0.95565111979281, -0.060335199665269, -0.23746618039481, 0.9695202302428, 0.05721341853462, -0.23822716645767, 0.96952279081058, 0.17387675788453, -0.23779376813896, 0.95562597123657, 0.28791894946238, -0.23611990382639, 0.92808947281902, 0.39787809969711, -0.23311086710279, 0.88732876737994, 0.50185275913732, -0.22902512950122, 0.83408111008655, 0.59862251876608, -0.2236545334909, 0.76917470686127, 0.686773878932, -0.217374749619, 0.69360641392995, 0.76487464264496, -0.2098064824764, 0.60905502292455, -0.77428391501926, -0.096920937778581, 0.62537248961125, -0.69399517455981, -0.10241840987289, 0.71265781901752, -0.60542381066016, -0.1065626786924, 0.78873722176377, -0.50784199110286, -0.11044193335186, 0.85434132021703, -0.40303941206181, -0.11355023114121, 0.90811099394989, -0.29224781379666, -0.11618984981496, 0.94925820203517, -0.17730850393948, -0.11818018158758, 0.97703384747442, -0.05979344668954, -0.11945876646463, 0.99103700579127, 0.058599484147317, -0.12012550929984, 0.99102773042591, 0.17602867679666, -0.12016568006294, 0.97702308789518, 0.29103556491225, -0.11945729715765, 0.94922508084858, 0.40164166306628, -0.11813458739915, 0.90814546948679, 0.50654658835759, -0.1161009061801, 0.85436007245625, 0.6039769086833, -0.11342935826947, 0.78888888600357, 0.69272882575311, -0.11015330733111, 0.71273629264598, 0.77216599030945, -0.10647805391823, 0.62643603619461, -0.77396376555512, 0.0048696532889419, 0.63321116231837, -0.69532114040719, 0.0049491471279569, 0.71868213950643, -0.6060652488246, 0.0038874421214649, 0.79540543244369, -0.50848113419124, 0.003152973956816, 0.8610673579499, -0.40327231980335, 0.0022351597728664, 0.91507728643061, -0.29230885986179, 0.0015531920751465, 0.95632270601543, -0.17701358969952, 0.00086524594042629, 0.98420802699996, -0.059243790234189, 0.00025344831443789, 0.99824351191613, 0.059339538824344, -0.00027720213746297, 0.99823781850373, 0.17709595048397, -0.00069804683999123, 0.98419334332883, 0.29231191822824, -0.0010481250720005, 0.95632245811523, 0.40325813838092, -0.0012406613127543, 0.91508542474955, 0.50831404482585, -0.0013552128798042, 0.86117071201408, 0.60598479889314, -0.0014148774052128, 0.79547496606265, 0.6950823121567, -0.0012685629584321, 0.71892904383876, 0.77370050087794, -0.0012419875981948, 0.63355030771679, -0.77153798942826, 0.10970186726151, 0.62665351765417, -0.69311613540803, 0.1118803014625, 0.71209045842625, -0.604108995172, 0.11444239959637, 0.78864140084509, -0.50661896975475, 0.11649391495788, 0.85426365207846, -0.40182328522893, 0.11831132656416, 0.90804211215915, -0.29110170929508, 0.11935129563119, 0.94921813250519, -0.17623186886245, 0.11996582536782, 0.97701101792204, -0.058752522305074, 0.12001097294189, 0.99103254613374, 0.059571368700493, 0.11962138434628, 0.99103076462753, 0.17700105394968, 0.11879071655417, 0.97701555389935, 0.29187824353126, 0.11747552529389, 0.9492136703135, 0.40250018154614, 0.11564632906167, 0.90808564047114, 0.50725033043003, 0.11335879231846, 0.85431076692426, 0.60465882495889, 0.1106937241044, 0.78875509814088, 0.69341318020326, 0.10752469873626, 0.71247217537394, 0.77232015803405, 0.10430138841494, 0.62661215585798, -0.7656078789582, 0.20993439712327, 0.60808891174039, -0.68737155863899, 0.21764189045682, 0.69293026192582, -0.59940805960742, 0.22351531783455, 0.7686032011194, -0.50252403441558, 0.22855540107114, 0.83380574684749, -0.39857822204048, 0.2323575135924, 0.88721214305949, -0.28889653735469, 0.23535155265392, 0.92798083890125, -0.17476619038261, 0.23722320635366, 0.95560553005225, -0.058278373328452, 0.23813513623158, 0.9694819689371, 0.059218573121852, 0.2378651268003, 0.96949128002767, 0.17571759366789, 0.23660479581708, 0.95558437506686, 0.28969970055527, 0.23418834134853, 0.928024732577, 0.39944328480979, 0.23083339548927, 0.8872209452821, 0.5033548659632, 0.22643684664922, 0.83388262566786, 0.5999411149941, 0.22111542760576, 0.76888141232206, 0.68790295561068, 0.21487662608973, 0.69326586474626, 0.76599792517135, 0.20781648409948, 0.60832514954564, -0.75410132535918, 0.30946319207273, 0.57927862367231, -0.67875793924457, 0.3196529274437, 0.66114269707011, -0.59168296071542, 0.32886701053214, 0.73604195762381, -0.4964411340665, 0.33679506446137, 0.80007204985628, -0.39376048426488, 0.34321427128352, 0.85273480345229, -0.28538753621772, 0.34801125331154, 0.89299614878235, -0.17278224378662, 0.35112971959241, 0.92024682354845, -0.057695285294946, 0.35265864829895, 0.9339716975555, 0.05820461615991, 0.35261796180349, 0.93395545700597, 0.17327263597523, 0.35079582670858, 0.92028195765539, 0.28574366554384, 0.3475384625288, 0.89306638872185, 0.39408542818382, 0.34251048305806, 0.85286765930507, 0.4966529861426, 0.33610484529866, 0.80023080690661, 0.59193836316241, 0.32811535000228, 0.7361720527903, 0.67862715121686, 0.31893775329357, 0.66162217250885, 0.75581104257595, 0.30750839926579, 0.57809017661715, -0.74114388364916, 0.40309551713724, 0.53686101347863, -0.66677994185866, 0.41705098254364, 0.61763499503699, -0.58141962865961, 0.42982615001232, 0.69079714545944, -0.48796906872363, 0.44008472760343, 0.75379812980612, -0.38702146528386, 0.4487073392926, 0.80552846571334, -0.2809813198152, 0.45533552773549, 0.84481894811655, -0.17026852925844, 0.45969852848374, 0.87159961613923, -0.057129731763082, 0.46204421296045, 0.88501488067627, 0.056598013366457, 0.46196320321514, 0.885091330744, 0.1696744941614, 0.45990509921984, 0.87160648560154, 0.28017934300592, 0.45548919257785, 0.84500244449206, 0.38648270316358, 0.44926185348741, 0.80547806125085, 0.48726725115971, 0.44062136227481, 0.75393861889039, 0.58063944526348, 0.4304488392141, 0.69106557678946, 0.66581113843193, 0.41769853886104, 0.618242232926, 0.74034333257333, 0.40593743576122, 0.53582324339452, -0.71839602841265, 0.49208827207521, 0.49168717580079, -0.65161981738977, 0.50756849636788, 0.56370722461201, -0.56888392603689, 0.52299441246673, 0.6347030197072, -0.47787677873614, 0.53748799482727, 0.69479525024379, -0.37924893367517, 0.54794678887351, 0.74560348904056, -0.27495355391929, 0.55606338514756, 0.7843430721855, -0.16689194396882, 0.56201382227019, 0.81011575877498, -0.057003011163135, 0.56425143776271, 0.82363278935522, 0.054792373152076, 0.56471734269075, 0.8234634896027, 0.16492652227631, 0.5621599106741, 0.81041685389726, 0.27284847937779, 0.55714360883347, 0.784311613096, 0.37691390789235, 0.54846959483533, 0.74640271273521, 0.47473752945377, 0.53951402193883, 0.69537680307839, 0.56679109341873, 0.52456138509254, 0.63528199226092, 0.64837920155748, 0.51273674936996, 0.56276588101378, 0.73190527205887, 0.47467946237535, 0.48887020847203]}}");
    aslam::Camera::Ptr sharedCamera = aslam::createCamera(node);
    CameraId id;
    generateId(&id);
    sharedCamera->setId(id);
    aslam::GenericCamera::Ptr genericCamera = std::dynamic_pointer_cast<aslam::GenericCamera>(sharedCamera);
    aslam::GenericCamera::UniquePtr uniqueCamera(new aslam::GenericCamera(*genericCamera));
    return std::move(uniqueCamera);
  }

  /// \brief Create a test camera object for unit testing. (without distortion)
  static GenericCamera::Ptr createTestCamera();

  /// \brief return the first value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> firstGridValue() const { return gridAccess(0,0); };
  /// \brief return the last value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> lastGridValue() const { return gridAccess(gridHeight()-1, gridWidth()-1); };

  // position of the pixel expressed in gridpoints
  Eigen::Vector2d transformImagePixelToGridPoint(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const;
  Eigen::Vector2d transformGridPointToImagePixel(const Eigen::Vector2d& gridpoint) const; // <- inverse of transformImagePixelToGridPoint

  double pixelScaleToGridScaleX(double length) const;
  double pixelScaleToGridScaleY(double length) const;

  Eigen::Vector3d gridAccess(const int y, const int x) const;
  Eigen::Vector3d gridAccess(const Eigen::Vector2d gridpoint) const;

  double getFocalLengthApproximation() const;
  void setFocalLengthApproximation();

 private:
  /// \brief Minimal depth for a valid projection.
  static const double kMinimumDepth;

  bool isValidImpl() const override;
  void setRandomImpl() override;
  bool isEqualImpl(const Sensor& other, const bool verbose) const override;


  bool isInCalibratedArea(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const;

  void interpolateCubicBSplineSurface(Eigen::Vector2d keypoint, Eigen::Vector3d* out_point_3d) const;
  void interpolateCubicBSpline(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c, Eigen::Vector3d d, double frac_y, Eigen::Vector3d* out_point_3d) const;

  bool loadFromYamlNodeImpl(const YAML::Node&) override;
  void saveToYamlNodeImpl(YAML::Node*) const override;
  Eigen::VectorXd getIntrinsics() const;
  Eigen::VectorXd getGrid() const;

  bool backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const;
  const ProjectionResult project3WithInitialEstimate(const Eigen::Ref<const Eigen::Vector3d>& point_3d, const Eigen::VectorXd* intrinsics,
                                  Eigen::Vector2d* out_keypoint) const;
  void CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(double frac_x, double frac_y, Eigen::Matrix<double, 3, 1> p[4][4], Eigen::Matrix<double, 3, 1>* result, Eigen::Matrix<double, 3, 2>* dresult_dxy) const;

  double focallengthApproximation;
};

}  // namespace aslam

#include "aslam/cameras/camera-generic-inl.h"

#endif  // ASLAM_CAMERAS_GENERIC_CAMERA_H_
