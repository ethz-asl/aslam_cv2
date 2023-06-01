#include <memory>
#include <utility>

#include <aslam/cameras/camera-generic.h>

#include <aslam/common/types.h>

#include "aslam/cameras/random-camera-generator.h"

namespace aslam {
std::ostream& operator<<(std::ostream& out, const GenericCamera& camera) {
  camera.printParameters(out, std::string(""));
  return out;
}

GenericCamera::GenericCamera()
    : Base(Eigen::Matrix<double, 1, 1>::Zero(), 0, 0, Camera::Type::kGeneric) {}

GenericCamera::GenericCamera(const Eigen::VectorXd& intrinsics,
                             uint32_t image_width, uint32_t image_height,
                             aslam::Distortion::UniquePtr& distortion)
  : GenericCamera(intrinsics, image_width, image_height){
  if(distortion->getType() != Distortion::Type::kNoDistortion){
    LOG(ERROR) << "Constructed Generic Camera with Distortion, distortion set to kNoDistortion";
  }
}

GenericCamera::GenericCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width,
                             uint32_t image_height)
    : Base(intrinsics, image_width, image_height, Camera::Type::kGeneric) {
  CHECK(intrinsicsValid(intrinsics));
}

bool GenericCamera::backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);
  if(!isInCalibratedArea(keypoint)){
    return false;
  }
  Eigen::Vector2d keypointGrid = transformImagePixelToGridPoint(keypoint);
  interpolateCubicBSplineSurface(keypointGrid, out_point_3d);
  *out_point_3d = (*out_point_3d).normalized();
  return true;
}

bool GenericCamera::backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const {
  CHECK_NOTNULL(out_point_3d);
  CHECK_NOTNULL(out_jacobian_pixel);
  if(!isInCalibratedArea(keypoint)){
    return false;
  }

  Eigen::Vector2d keypoint_grid = transformImagePixelToGridPoint(keypoint, intrinsics);
  Eigen::Vector2d upperleft_grid = (keypoint_grid - Eigen::Vector2d(2,2)).array().ceil();
  Eigen::Vector2d frac = keypoint_grid - upperleft_grid + Eigen::Vector2d(2,2);

  Eigen::Vector3d gridInterpolationWindow[4][4];
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      // gridInterpolationWindow[i][j] = gridAccess(floor_y - 3 + i, floor_x - 3 + j);
      gridInterpolationWindow[i][j] = intrinsics.segment<3>(6 + 3*((upperleft_grid.y() + i)*intrinsics(Parameters::kGridWidth) + (upperleft_grid.x() + j)));
    }
  }

  CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(frac.x(), frac.y(), gridInterpolationWindow, out_point_3d, out_jacobian_pixel);

  (*out_jacobian_pixel)(0,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(0,0));
  (*out_jacobian_pixel)(0,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(0,1));
  (*out_jacobian_pixel)(1,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(1,0));
  (*out_jacobian_pixel)(1,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(1,1));
  (*out_jacobian_pixel)(2,0) = pixelScaleToGridScaleX((*out_jacobian_pixel)(2,0));
  (*out_jacobian_pixel)(2,1) = pixelScaleToGridScaleY((*out_jacobian_pixel)(2,1));

  return true;
}

const ProjectionResult GenericCamera::project3WithInitialEstimate(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics,
    Eigen::Vector2d* out_keypoint) const {

  // optimize projected position using the Levenberg-Marquardt method
  double epsilon = 1e-12;
  int maxIterations = 100;
  Eigen::Vector3d pointDirection = point_3d.normalized();

  double lambda = -1.0;
  for(int i = 0; i < maxIterations; i++){
    Eigen::Matrix<double, 3, 2> ddxy_dxy;

    Eigen::Vector3d bearingVector;
    // unproject with jacobian
    if(!backProject3WithJacobian(*out_keypoint, *intrinsics, &bearingVector, &ddxy_dxy)){
      return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
    }

    Eigen::Vector3d diff = bearingVector - pointDirection;
    double cost = diff.squaredNorm();

    double H_0_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 0) + ddxy_dxy(1, 0) * ddxy_dxy(1, 0) + ddxy_dxy(2, 0) * ddxy_dxy(2, 0);
    double H_0_1_and_1_0 = ddxy_dxy(0, 0) * ddxy_dxy(0, 1) + ddxy_dxy(1, 0) * ddxy_dxy(1, 1) + ddxy_dxy(2, 0) * ddxy_dxy(2, 1);
    double H_1_1 = ddxy_dxy(0, 1) * ddxy_dxy(0, 1) + ddxy_dxy(1, 1) * ddxy_dxy(1, 1) + ddxy_dxy(2, 1) * ddxy_dxy(2, 1);
    double b_0 = diff.x() * ddxy_dxy(0, 0) + diff.y() * ddxy_dxy(1, 0) + diff.z() * ddxy_dxy(2, 0);
    double b_1 = diff.x() * ddxy_dxy(0, 1) + diff.y() * ddxy_dxy(1, 1) + diff.z() * ddxy_dxy(2, 1);
  
    // TODO: change to if i == 0
    if(lambda < 0){
      constexpr double initialLambdaFactor = 0.01;
      lambda = initialLambdaFactor * 0.5 * (H_0_0 + H_1_1);
    }

    bool updateAccepted = false;
    for(int lmIteration = 0; lmIteration < 10; lmIteration++){
      double H_0_0_lm = H_0_0 + lambda;
      double H_1_1_lm = H_1_1 + lambda;

      // Solve the linear system
      double x_1 = (b_1 - H_0_1_and_1_0 / H_0_0_lm * b_0) /
                    (H_1_1_lm - H_0_1_and_1_0 * H_0_1_and_1_0 / H_0_0_lm);
      double x_0 = (b_0 - H_0_1_and_1_0 * x_1) / H_0_0_lm;

      Eigen::Vector2d testResult(
        std::max(calibrationMinX(), std::min(calibrationMaxX() + 0.999, out_keypoint->x() - x_0)),
        std::max(calibrationMinY(), std::min(calibrationMaxY() + 0.999, out_keypoint->y() - x_1))
      );

      double testCost = std::numeric_limits<double>::infinity();

      Eigen::Vector3d testDirection;
      if(backProject3(testResult, &testDirection)){
        Eigen::Vector3d testDiff = testDirection - pointDirection;
        testCost = testDiff.squaredNorm();
      }

      if(testCost < cost){
        lambda *= 0.5;
        *out_keypoint = testResult;
        updateAccepted = true;
        break;
      } else {
        lambda *= 2.;
      }
    }

    if(cost < epsilon){
      return evaluateProjectionResult(*out_keypoint, point_3d);
    }
    if(!updateAccepted){
      return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
    }
  }
  // not found in maxinterations -> bearing vector not found
  return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}


const ProjectionResult GenericCamera::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const {
  CHECK_NOTNULL(out_keypoint);

  if(distortion_coefficients_external){
    LOG(FATAL) << "External distortion coefficients provided for projection with generic model. Generic models have distortion " 
    << "already included in the model, abort";
  }
  if(out_jacobian_intrinsics){
    LOG(FATAL) << "Jacobian for intrinsics can't be calculated for generic models, abort";
  }
  if(out_jacobian_distortion){
    LOG(FATAL) << "Jacobian for distortion can't be calculated for generic models, distortion is included in model, abort";
  }

  const Eigen::VectorXd* intrinsics;
  if (!intrinsics_external)
    intrinsics = &getParameters();
  else
    intrinsics = intrinsics_external;

  // initialize the projected position in the center of the calibrated area
  *out_keypoint = centerOfCalibratedArea();

  ProjectionResult result = project3WithInitialEstimate(point_3d, intrinsics, out_keypoint);

  if(out_jacobian_point != nullptr && !(result == ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID))){

    const double numericalDiffDelta = 1e-4;
    for( int dimension = 0; dimension < 3; dimension++){

      Eigen::Vector3d offsetPoint = point_3d;
      offsetPoint(dimension) += numericalDiffDelta;
      Eigen::Vector2d offsetPixel = *out_keypoint;
      if(project3WithInitialEstimate(offsetPoint, intrinsics, &offsetPixel) == ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID)){
        return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
      }
      (*out_jacobian_point)(0, dimension) = (offsetPixel.x() - out_keypoint->x()) / numericalDiffDelta;
      (*out_jacobian_point)(1, dimension) = (offsetPixel.y() - out_keypoint->y()) / numericalDiffDelta;
    }
  }
  return result;
}

Eigen::Vector2d GenericCamera::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom(); // [-1 : 1]
  out += Eigen::Vector2d(1, 1); // [0 : 2]
  out *= 0.5; // [0 : 1]
  
  out(0) = calibrationMinX() + out(0) * (calibrationMaxX() - calibrationMinX());
  out(1) = calibrationMinY() + out(1) * (calibrationMaxY() - calibrationMinY());
  return out;
}

Eigen::Vector3d GenericCamera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

bool GenericCamera::areParametersValid(const Eigen::VectorXd& parameters) {
  return (parameters[0] >= 0.0) &&
        (parameters[1] >= 0.0) &&
        (parameters[2] >= 0.0) &&
        (parameters[3] >= 0.0) &&
        (parameters[0] < parameters[2]) &&
        (parameters[1] < parameters[3]) &&
        (parameters[4] >= 4.0) &&
        (parameters[5] >= 4.0) &&
        (parameters.size() - 6 == parameters[4]*parameters[5]*3);;
}

bool GenericCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) const {
  return areParametersValid(intrinsics);
}

void GenericCamera::printParameters(std::ostream& out, const std::string& text) const {
  Camera::printParameters(out, text);
  out << "  minCalibration (x,y): "
      << calibrationMinX() << ", " << calibrationMinY() << std::endl;
  out << "  maxCalibration (x,y): "
      << calibrationMaxX() << ", " << calibrationMaxY() << std::endl;
  out << "  gridWidth, gridHeight (x,y): "
      << gridWidth() << ", " << gridHeight() << std::endl;
  out << "  gridvector at gridpoint (0,0), (x,y,z): "
      << firstGridValue().transpose() << std::endl;
}
const double GenericCamera::kMinimumDepth = 1e-10;

bool GenericCamera::isValidImpl() const {
  return intrinsicsValid(intrinsics_);
}

void GenericCamera::setRandomImpl() {
  GenericCamera::Ptr test_camera = GenericCamera::createTestCamera();
  CHECK(test_camera);
  line_delay_nanoseconds_ = test_camera->line_delay_nanoseconds_;
  image_width_ = test_camera->image_width_;
  image_height_ = test_camera->image_height_;
  mask_= test_camera->mask_;
  intrinsics_ = test_camera->intrinsics_;
  camera_type_ = test_camera->camera_type_;
  if (test_camera->distortion_) {
    distortion_ = std::move(test_camera->distortion_);
  }
}

bool GenericCamera::isEqualImpl(const Sensor& other, const bool verbose) const {
  const GenericCamera* other_camera =
      dynamic_cast<const GenericCamera*>(&other);
  if (other_camera == nullptr) {
    return false;
  }

  // Verify that the base members are equal.
  if (!isEqualCameraImpl(*other_camera, verbose)) {
    return false;
  }

  // Compare the distortion model (if distortion is set for both).
  if (!(*(this->distortion_) == *(other_camera->distortion_))) {
    return false;
  }

  return true;
}

GenericCamera::Ptr GenericCamera::createTestCamera() {
  YAML::Node node = YAML::Load("{type: generic, intrinsics: {rows: 6, cols: 1, data: [15,15,736,464,16,11]}, image_height: 480, image_width: 752,  grid: {rows: 176, cols: 3, data: [-0.73680864697228, -0.48562625042812, 0.47040425448969, -0.65643352034764, -0.49748019131363, 0.56710536288644, -0.57529862830316, -0.51573877806511, 0.63485825274026, -0.4829821466507, -0.53015078136865, 0.69689912830401, -0.38444374137903, -0.54243549487623, 0.74697171540352, -0.28086806363103, -0.55134648666921, 0.78557633777353, -0.1721103096518, -0.55960036817312, 0.810694436426, -0.061722237678026, -0.56287084396511, 0.82423708870083, 0.04989935551493, -0.56445602355111, 0.82395354953788, 0.16062533537446, -0.56247278237593, 0.81106341966712, 0.26901089536198, -0.55820951520686, 0.78487914694497, 0.37276222195738, -0.55095630336118, 0.74665619776975, 0.47204414033422, -0.54025171159243, 0.6966365032767, 0.56275855431752, -0.52822827676352, 0.6358283551163, 0.64784236585731, -0.51131660361745, 0.5646730026002, 0.71758270837233, -0.50356690402212, 0.48113971964348, -0.74482660879238, -0.3909445771263, 0.54073622076792, -0.67189174532953, -0.40740587055911, 0.61853208420583, -0.58631871071313, -0.42173191952584, 0.69164482035272, -0.49253828501389, -0.43359083149535, 0.75458931124072, -0.39178761654292, -0.44396537591327, 0.80585185270855, -0.28515926095472, -0.45196677267159, 0.84522791736465, -0.17454731383209, -0.45740159278634, 0.87196159213036, -0.061229766592886, -0.46096098857752, 0.8853055307026, 0.052817721847604, -0.46193577048946, 0.88533927519406, 0.16604198640003, -0.46064826643592, 0.87191354696547, 0.27681523432077, -0.45693512124914, 0.84533048035486, 0.38346886842054, -0.45093315196242, 0.80598444117334, 0.48430956889612, -0.44281560227242, 0.75455853574111, 0.57829427650963, -0.43228773251534, 0.69188369403603, 0.66344720274583, -0.42005923335663, 0.61918337319454, 0.74000379826004, -0.40493021217735, 0.53705297860334, -0.75990179436045, -0.29680790462239, 0.57832026653182, -0.68243718745857, -0.31039318190435, 0.66176699660902, -0.5953610787439, -0.32079809410701, 0.73663679566949, -0.49982352918459, -0.33047927303425, 0.80059970632533, -0.39729683367312, -0.33831840695849, 0.8530509254824, -0.28865961220064, -0.3445488781225, 0.89328701931054, -0.17603626334314, -0.34914465656895, 0.92038537731623, -0.060806842321716, -0.35179614609808, 0.93409956616915, 0.055267105910384, -0.35278842202231, 0.93406952433494, 0.17053574142323, -0.35186586124566, 0.92038468945712, 0.28323547455551, -0.34918832863142, 0.89322179614075, 0.39168807991485, -0.34473134068351, 0.85307722440772, 0.49450477832578, -0.33842753159012, 0.80058218196184, 0.58997181479829, -0.33060847079331, 0.73663511780481, 0.67706344707325, -0.32104402890598, 0.66220527039665, 0.75420409067053, -0.31050005146438, 0.57858958481505, -0.76812264873815, -0.2009072021158, 0.60796701607364, -0.68992177844774, -0.20759326624913, 0.69347889328481, -0.60163662553573, -0.21557203106791, 0.76913072376238, -0.50503800010051, -0.22209778457673, 0.83403188940267, -0.40098308456185, -0.22779772730831, 0.88731097216727, -0.29110727643508, -0.23216340458488, 0.92809304877157, -0.17695562751952, -0.23540952216918, 0.95565111979281, -0.060335199665269, -0.23746618039481, 0.9695202302428, 0.05721341853462, -0.23822716645767, 0.96952279081058, 0.17387675788453, -0.23779376813896, 0.95562597123657, 0.28791894946238, -0.23611990382639, 0.92808947281902, 0.39787809969711, -0.23311086710279, 0.88732876737994, 0.50185275913732, -0.22902512950122, 0.83408111008655, 0.59862251876608, -0.2236545334909, 0.76917470686127, 0.686773878932, -0.217374749619, 0.69360641392995, 0.76487464264496, -0.2098064824764, 0.60905502292455, -0.77428391501926, -0.096920937778581, 0.62537248961125, -0.69399517455981, -0.10241840987289, 0.71265781901752, -0.60542381066016, -0.1065626786924, 0.78873722176377, -0.50784199110286, -0.11044193335186, 0.85434132021703, -0.40303941206181, -0.11355023114121, 0.90811099394989, -0.29224781379666, -0.11618984981496, 0.94925820203517, -0.17730850393948, -0.11818018158758, 0.97703384747442, -0.05979344668954, -0.11945876646463, 0.99103700579127, 0.058599484147317, -0.12012550929984, 0.99102773042591, 0.17602867679666, -0.12016568006294, 0.97702308789518, 0.29103556491225, -0.11945729715765, 0.94922508084858, 0.40164166306628, -0.11813458739915, 0.90814546948679, 0.50654658835759, -0.1161009061801, 0.85436007245625, 0.6039769086833, -0.11342935826947, 0.78888888600357, 0.69272882575311, -0.11015330733111, 0.71273629264598, 0.77216599030945, -0.10647805391823, 0.62643603619461, -0.77396376555512, 0.0048696532889419, 0.63321116231837, -0.69532114040719, 0.0049491471279569, 0.71868213950643, -0.6060652488246, 0.0038874421214649, 0.79540543244369, -0.50848113419124, 0.003152973956816, 0.8610673579499, -0.40327231980335, 0.0022351597728664, 0.91507728643061, -0.29230885986179, 0.0015531920751465, 0.95632270601543, -0.17701358969952, 0.00086524594042629, 0.98420802699996, -0.059243790234189, 0.00025344831443789, 0.99824351191613, 0.059339538824344, -0.00027720213746297, 0.99823781850373, 0.17709595048397, -0.00069804683999123, 0.98419334332883, 0.29231191822824, -0.0010481250720005, 0.95632245811523, 0.40325813838092, -0.0012406613127543, 0.91508542474955, 0.50831404482585, -0.0013552128798042, 0.86117071201408, 0.60598479889314, -0.0014148774052128, 0.79547496606265, 0.6950823121567, -0.0012685629584321, 0.71892904383876, 0.77370050087794, -0.0012419875981948, 0.63355030771679, -0.77153798942826, 0.10970186726151, 0.62665351765417, -0.69311613540803, 0.1118803014625, 0.71209045842625, -0.604108995172, 0.11444239959637, 0.78864140084509, -0.50661896975475, 0.11649391495788, 0.85426365207846, -0.40182328522893, 0.11831132656416, 0.90804211215915, -0.29110170929508, 0.11935129563119, 0.94921813250519, -0.17623186886245, 0.11996582536782, 0.97701101792204, -0.058752522305074, 0.12001097294189, 0.99103254613374, 0.059571368700493, 0.11962138434628, 0.99103076462753, 0.17700105394968, 0.11879071655417, 0.97701555389935, 0.29187824353126, 0.11747552529389, 0.9492136703135, 0.40250018154614, 0.11564632906167, 0.90808564047114, 0.50725033043003, 0.11335879231846, 0.85431076692426, 0.60465882495889, 0.1106937241044, 0.78875509814088, 0.69341318020326, 0.10752469873626, 0.71247217537394, 0.77232015803405, 0.10430138841494, 0.62661215585798, -0.7656078789582, 0.20993439712327, 0.60808891174039, -0.68737155863899, 0.21764189045682, 0.69293026192582, -0.59940805960742, 0.22351531783455, 0.7686032011194, -0.50252403441558, 0.22855540107114, 0.83380574684749, -0.39857822204048, 0.2323575135924, 0.88721214305949, -0.28889653735469, 0.23535155265392, 0.92798083890125, -0.17476619038261, 0.23722320635366, 0.95560553005225, -0.058278373328452, 0.23813513623158, 0.9694819689371, 0.059218573121852, 0.2378651268003, 0.96949128002767, 0.17571759366789, 0.23660479581708, 0.95558437506686, 0.28969970055527, 0.23418834134853, 0.928024732577, 0.39944328480979, 0.23083339548927, 0.8872209452821, 0.5033548659632, 0.22643684664922, 0.83388262566786, 0.5999411149941, 0.22111542760576, 0.76888141232206, 0.68790295561068, 0.21487662608973, 0.69326586474626, 0.76599792517135, 0.20781648409948, 0.60832514954564, -0.75410132535918, 0.30946319207273, 0.57927862367231, -0.67875793924457, 0.3196529274437, 0.66114269707011, -0.59168296071542, 0.32886701053214, 0.73604195762381, -0.4964411340665, 0.33679506446137, 0.80007204985628, -0.39376048426488, 0.34321427128352, 0.85273480345229, -0.28538753621772, 0.34801125331154, 0.89299614878235, -0.17278224378662, 0.35112971959241, 0.92024682354845, -0.057695285294946, 0.35265864829895, 0.9339716975555, 0.05820461615991, 0.35261796180349, 0.93395545700597, 0.17327263597523, 0.35079582670858, 0.92028195765539, 0.28574366554384, 0.3475384625288, 0.89306638872185, 0.39408542818382, 0.34251048305806, 0.85286765930507, 0.4966529861426, 0.33610484529866, 0.80023080690661, 0.59193836316241, 0.32811535000228, 0.7361720527903, 0.67862715121686, 0.31893775329357, 0.66162217250885, 0.75581104257595, 0.30750839926579, 0.57809017661715, -0.74114388364916, 0.40309551713724, 0.53686101347863, -0.66677994185866, 0.41705098254364, 0.61763499503699, -0.58141962865961, 0.42982615001232, 0.69079714545944, -0.48796906872363, 0.44008472760343, 0.75379812980612, -0.38702146528386, 0.4487073392926, 0.80552846571334, -0.2809813198152, 0.45533552773549, 0.84481894811655, -0.17026852925844, 0.45969852848374, 0.87159961613923, -0.057129731763082, 0.46204421296045, 0.88501488067627, 0.056598013366457, 0.46196320321514, 0.885091330744, 0.1696744941614, 0.45990509921984, 0.87160648560154, 0.28017934300592, 0.45548919257785, 0.84500244449206, 0.38648270316358, 0.44926185348741, 0.80547806125085, 0.48726725115971, 0.44062136227481, 0.75393861889039, 0.58063944526348, 0.4304488392141, 0.69106557678946, 0.66581113843193, 0.41769853886104, 0.618242232926, 0.74034333257333, 0.40593743576122, 0.53582324339452, -0.71839602841265, 0.49208827207521, 0.49168717580079, -0.65161981738977, 0.50756849636788, 0.56370722461201, -0.56888392603689, 0.52299441246673, 0.6347030197072, -0.47787677873614, 0.53748799482727, 0.69479525024379, -0.37924893367517, 0.54794678887351, 0.74560348904056, -0.27495355391929, 0.55606338514756, 0.7843430721855, -0.16689194396882, 0.56201382227019, 0.81011575877498, -0.057003011163135, 0.56425143776271, 0.82363278935522, 0.054792373152076, 0.56471734269075, 0.8234634896027, 0.16492652227631, 0.5621599106741, 0.81041685389726, 0.27284847937779, 0.55714360883347, 0.784311613096, 0.37691390789235, 0.54846959483533, 0.74640271273521, 0.47473752945377, 0.53951402193883, 0.69537680307839, 0.56679109341873, 0.52456138509254, 0.63528199226092, 0.64837920155748, 0.51273674936996, 0.56276588101378, 0.73190527205887, 0.47467946237535, 0.48887020847203]}}");
  aslam::Camera::Ptr camera = aslam::createCamera(node);
  return std::dynamic_pointer_cast<aslam::GenericCamera>(camera);
}

bool GenericCamera::isInCalibratedArea(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const {
  double x = keypoint.x();
  double y = keypoint.y();
  return x >= calibrationMinX() && y >= calibrationMinY() && x < calibrationMaxX() + 1 && y < calibrationMaxY() + 1;
}

Eigen::Vector2d GenericCamera::transformImagePixelToGridPoint(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const {
  return Eigen::Vector2d(
    1. + (gridWidth() - 3.) * (keypoint.x() - calibrationMinX()) / (calibrationMaxX() - calibrationMinX() + 1.),
    1. + (gridHeight() - 3.) * (keypoint.y() - calibrationMinY()) / (calibrationMaxY() - calibrationMinY() + 1.)
  );
}

Eigen::Vector2d GenericCamera::transformImagePixelToGridPoint(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics) const {
  return Eigen::Vector2d(
    1. + (intrinsics(Parameters::kGridWidth) - 3.) * (keypoint.x() - intrinsics(Parameters::kCalibrationMinX)) / (intrinsics(Parameters::kCalibrationMaxX) - intrinsics(Parameters::kCalibrationMinX) + 1.),
    1. + (intrinsics(Parameters::kGridHeight) - 3.) * (keypoint.y() - intrinsics(Parameters::kCalibrationMinY)) / (intrinsics(Parameters::kCalibrationMaxY) - intrinsics(Parameters::kCalibrationMinY) + 1.)
  );
}

Eigen::Vector2d GenericCamera::transformGridPointToImagePixel(const Eigen::Vector2d& gridpoint) const {
  return Eigen::Vector2d(
    calibrationMinX() + (gridpoint.x()-1.)*(calibrationMaxX() - calibrationMinX() + 1.)/(gridWidth() - 3.),
    calibrationMinY() + (gridpoint.y()-1.)*(calibrationMaxY() - calibrationMinY() + 1.)/(gridHeight() - 3.)
  );
}

double GenericCamera::pixelScaleToGridScaleX(double length) const {
  return length *((gridWidth() - 3.) / (calibrationMaxX() - calibrationMinX() + 1.));
}

double GenericCamera::pixelScaleToGridScaleY(double length) const {
  return length *((gridHeight() - 3.) / (calibrationMaxY() - calibrationMinY() + 1.));
}


/*Eigen::Vector3d GenericCamera::valueAtGridpoint(const int x, const int y) const {
  //Eigen::Matrix<double, Eigen::Dynamic, 3> grid = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3>>(const_cast<double*>(intrinsics_.data() + 6), gridWidth() * gridHeight(), 3);
  //return grid.row(y * gridWidth() + x);
  return intrinsics_.segment<3>(6 + 3*(y*gridWidth() + x));
}
*/
Eigen::Vector3d GenericCamera::gridAccess(const int row, const int col) const {
  //Eigen::Matrix<double, Eigen::Dynamic, 3> grid = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3>>(const_cast<double*>(intrinsics_.data() + 6), gridWidth() * gridHeight(), 3);
  //return grid.row(y * gridWidth() + x);
  return intrinsics_.segment<3>(6 + 3*(row*gridWidth() + col));
}

Eigen::Vector3d GenericCamera::gridAccess(const Eigen::Vector2d gridpoint) const {
  //Eigen::Matrix<double, Eigen::Dynamic, 3> grid = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3>>(const_cast<double*>(intrinsics_.data() + 6), gridWidth() * gridHeight(), 3);
  //return grid.row(y * gridWidth() + x);
  return intrinsics_.segment<3>(6 + 3*(gridpoint.y()*gridWidth() + gridpoint.x()));
}

double GenericCamera::getFocalLengthApproximation() const {
  return focallengthApproximation;
}

void GenericCamera::setFocalLengthApproximation(){


  const int stepsize = 1;
  const int numCalibratedPixels = ((calibrationMaxX() - calibrationMinX()) / stepsize) * ((calibrationMaxY() - calibrationMinY()) / stepsize);

  // set A and b s.t. Ax = b solves fu * direction[0] + cu = keypoint[0] for x = [fu, cu]
  Eigen::Matrix<double, Eigen::Dynamic, 2> A_u;
  A_u.resize(numCalibratedPixels, 2);
  Eigen::VectorXd b_u;
  b_u.resize(numCalibratedPixels);

  // set A and b s.t. Ax = b solves fv * direction[1] + cv = keypoint[1] for x = [fv, cv]
  Eigen::Matrix<double, Eigen::Dynamic, 2> A_v;
  A_v.resize(numCalibratedPixels, 2);
  Eigen::VectorXd b_v;
  b_v.resize(numCalibratedPixels);

  int counter = 0;
  for(int i = calibrationMinX(); i < calibrationMaxX(); i += stepsize){
    for(int j = calibrationMinY(); j < calibrationMaxY(); j += stepsize){

      Eigen::Vector3d direction;
      Eigen::Vector2d keypoint = Eigen::Vector2d(i, j);
      bool backProjectWorked = backProject3(keypoint, &direction);
      if(!backProjectWorked){
        LOG(FATAL) << "backProject3 didn't work for keypoint: " << i << ", " << j;
      }

      // z-value has to be = 1, like in pinhole model
      direction = direction / direction(2);

      A_u.row(counter) = Eigen::Vector2d(direction(0), 1.0);
      b_u(counter) = keypoint(0);
      A_v.row(counter) = Eigen::Vector2d(direction(1), 1.0);
      b_v(counter) = keypoint(1);

      counter++;
    }
  }

  // solve least squares Ax = b using QR decomposition
  Eigen::Vector2d fu_cu = A_u.colPivHouseholderQr().solve(b_u);
  Eigen::Vector2d fv_cv = A_v.colPivHouseholderQr().solve(b_v);

  focallengthApproximation = fu_cu(0)+ fv_cv(0);
}


void GenericCamera::interpolateCubicBSplineSurface(Eigen::Vector2d keypoint, Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  double x = keypoint.x();
  double y = keypoint.y();

  // start at bottom right corner of used 4x4 grid
  x+=2.0;
  y+=2.0;

  double floor_x = std::floor(x);
  double floor_y = std::floor(y);

  double frac_x = x - (floor_x - 3);
  double frac_y = y - (floor_y - 3);

  // i == 0
  double a_coeff = -1./6. * (frac_x - 4) * (frac_x - 4) * (frac_x - 4);

  // i == 1
  double b_coeff = 1./2. * frac_x * frac_x * frac_x - 11./2. * frac_x * frac_x + (39./2.) * frac_x - 131./6.;

  // i == 2
  double c_coeff = -1./2. * frac_x * frac_x * frac_x + 5 * frac_x * frac_x - 16 * frac_x + 50./3.;

  // i == 3
  double d_coeff = 1./6. * (frac_x - 3) * (frac_x - 3) * (frac_x - 3);


  int row0 = floor_y - 3;
  Eigen::Matrix<double, 3, 1> a = a_coeff * gridAccess(row0, floor_x - 3) + b_coeff * gridAccess(row0, floor_x - 2) 
                              + c_coeff * gridAccess(row0, floor_x - 1) + d_coeff * gridAccess(row0, floor_x);
  int row1 = floor_y - 2;
  Eigen::Matrix<double, 3, 1> b = a_coeff * gridAccess(row1, floor_x - 3) + b_coeff * gridAccess(row1, floor_x - 2) 
                              + c_coeff * gridAccess(row1, floor_x - 1) + d_coeff * gridAccess(row1, floor_x);
  int row2 = floor_y - 1;
  Eigen::Matrix<double, 3, 1> c = a_coeff * gridAccess(row2, floor_x - 3) + b_coeff * gridAccess(row2, floor_x - 2) 
                              + c_coeff * gridAccess(row2, floor_x - 1) + d_coeff * gridAccess(row2, floor_x);
  int row3 = floor_y;
  Eigen::Matrix<double, 3, 1> d = a_coeff * gridAccess(row3, floor_x - 3) + b_coeff * gridAccess(row3, floor_x - 2) 
                              + c_coeff * gridAccess(row3, floor_x - 1) + d_coeff * gridAccess(row3, floor_x);
  interpolateCubicBSpline(a, b, c, d, frac_y, out_point_3d);
}


void GenericCamera::interpolateCubicBSpline(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c, Eigen::Vector3d d, double frac_y, Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  // i == 0
  double a_coeff = -1./6. * (frac_y - 4) * (frac_y - 4) * (frac_y - 4);

  // i == 1
  double b_coeff = 1./2. * frac_y * frac_y * frac_y - 11./2. * frac_y * frac_y + (39./2.) * frac_y - 131./6.;

  // i == 2
  double c_coeff = -1./2. * frac_y * frac_y * frac_y + 5 * frac_y * frac_y - 16 * frac_y + 50./3.;

  // i == 3
  double d_coeff = 1./6. * (frac_y - 3) * (frac_y - 3) * (frac_y - 3);

  *out_point_3d = a_coeff * a + b_coeff * b + c_coeff * c + d_coeff * d;
}

bool GenericCamera::loadFromYamlNodeImpl(const YAML::Node& yaml_node) {
  if (!yaml_node.IsMap()) {
    LOG(ERROR) << "Unable to parse the camera because the node is not a map.";
    return false;
  }
  LOG(ERROR) << "########## Using generic camera model ##########";
  // Distortion is directly included in the generic camera model
  const YAML::Node& distortion_config = yaml_node["distortion"];
  if(distortion_config.IsDefined() && !distortion_config.IsNull()) {
    if(!distortion_config.IsMap()) {
      LOG(ERROR) << "Unable to parse the camera because the distortion node is "
                    "not a map.";
      return false;
    }
    std::string distortion_type;
    if(YAML::safeGet(distortion_config, "type", &distortion_type)){
      if(distortion_type != "none"){
        LOG(ERROR) << "Distortion is provided for generic model, which have the distortion " 
        << "included in the model. Distortion set to NullDistortion.";
      }
    }
  }
  distortion_.reset(new aslam::NullDistortion());

  // Get the image width and height
  if (!YAML::safeGet(yaml_node, "image_width", &image_width_) ||
      !YAML::safeGet(yaml_node, "image_height", &image_height_)) {
    LOG(ERROR) << "Unable to get the image size.";
    return false;
  }

  // Get the camera type
  std::string camera_type;
  if (!YAML::safeGet(yaml_node, "type", &camera_type)) {
    LOG(ERROR) << "Unable to get camera type";
    return false;
  }
  if(camera_type != "generic"){
     LOG(ERROR) << "Camera model: \"" << camera_type << "\", but generic expected";
    return false;
  }
  camera_type_ = Type::kGeneric;

  // Get the camera intrinsics
  Eigen::VectorXd tempIntrinsics;
  if (!YAML::safeGet(yaml_node, "intrinsics", &tempIntrinsics)) {
    LOG(ERROR) << "Unable to get intrinsics";
    return false;
  }

  // Get the grid for the generic model and normalize the vectors
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> tempGrid;
  if (!YAML::safeGet(yaml_node, "grid", &tempGrid)) {
    LOG(ERROR) << "Unable to get grid";
    return false;
  }
  tempGrid.rowwise().normalize();

  // Concatenate tempIntrinsics and tempGrid into one intrinsics_ vector
  intrinsics_.resize(6 + tempGrid.size());
  intrinsics_.head<6>() = tempIntrinsics; 
  intrinsics_.tail(intrinsics_.size() - 6) = Eigen::Map<Eigen::VectorXd>(tempGrid.data(), tempGrid.size());

  if (!intrinsicsValid(intrinsics_)) {
    LOG(ERROR) << "Invalid intrinsics parameters for the " << camera_type
               << " camera model" << tempIntrinsics.transpose() << std::endl;
    return false;
  }

  // calculate approx. focal length only once
  setFocalLengthApproximation();

  // Get the optional linedelay in nanoseconds or set the default
  if (!YAML::safeGet(
          yaml_node, "line-delay-nanoseconds", &line_delay_nanoseconds_)) {
    LOG(WARNING) << "Unable to parse parameter line-delay-nanoseconds."
                 << "Setting to default value = 0.";
    line_delay_nanoseconds_ = 0;
  }

  // Get the optional compressed definition for images or set the default
  if (YAML::hasKey(yaml_node, "compressed")) {
    if (!YAML::safeGet(yaml_node, "compressed", &is_compressed_)) {
      LOG(WARNING) << "Unable to parse parameter compressed."
                   << "Setting to default value = false.";
      is_compressed_ = false;
    }
  }

  return true;
}

void GenericCamera::saveToYamlNodeImpl(YAML::Node* yaml_node) const { 
  CHECK_NOTNULL(yaml_node);
  YAML::Node& node = *yaml_node;

  node["compressed"] = hasCompressedImages();
  node["line-delay-nanoseconds"] = getLineDelayNanoSeconds();
  node["image_height"] = imageHeight();
  node["image_width"] = imageWidth();
  node["type"] = "generic";
  node["intrinsics"] = getIntrinsics();
  node["grid"] = getGrid();
}

Eigen::VectorXd GenericCamera::getIntrinsics() const{
  return intrinsics_.head(6);
}

Eigen::VectorXd GenericCamera::getGrid() const {
  return intrinsics_.tail(intrinsics_.size() - 6);
}

void GenericCamera::CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(double frac_x, double frac_y, Eigen::Matrix<double, 3, 1> p[4][4], Eigen::Matrix<double, 3, 1>* result, Eigen::Matrix<double, 3, 2>* dresult_dxy) const {
  const double term0 = 0.166666666666667*frac_y;
  const double term1 = -term0 + 0.666666666666667;
  const double term2 = (frac_y - 4) * (frac_y - 4);
  const double term3 = (frac_x - 4) * (frac_x - 4);
  const double term4 = 0.166666666666667*frac_x;
  const double term5 = -term4 + 0.666666666666667;
  const double term6 = p[0][0].x()*term5;
  const double term7 = (frac_x - 3) * (frac_x - 3);
  const double term8 = term4 - 0.5;
  const double term9 = p[0][3].x()*term8;
  const double term10 = frac_x * frac_x;
  const double term11 = 0.5*frac_x*term10;
  const double term12 = 19.5*frac_x - 5.5*term10 + term11 - 21.8333333333333;
  const double term13 = -16*frac_x + 5*term10 - term11 + 16.6666666666667;
  const double term14 = p[0][1].x()*term12 + p[0][2].x()*term13 + term3*term6 + term7*term9;
  const double term15 = term14*term2;
  const double term16 = term1*term15;
  const double term17 = term0 - 0.5;
  const double term18 = (frac_y - 3) * (frac_y - 3);
  const double term19 = p[3][0].x()*term5;
  const double term20 = p[3][3].x()*term8;
  const double term21 = p[3][1].x()*term12 + p[3][2].x()*term13 + term19*term3 + term20*term7;
  const double term22 = term18*term21;
  const double term23 = term17*term22;
  const double term24 = frac_y * frac_y;
  const double term25 = 0.5*frac_y*term24;
  const double term26 = -16*frac_y + 5*term24 - term25 + 16.6666666666667;
  const double term27 = p[2][0].x()*term5;
  const double term28 = p[2][3].x()*term8;
  const double term29 = p[2][1].x()*term12 + p[2][2].x()*term13 + term27*term3 + term28*term7;
  const double term30 = term26*term29;
  const double term31 = 19.5*frac_y - 5.5*term24 + term25 - 21.8333333333333;
  const double term32 = p[1][0].x()*term5;
  const double term33 = p[1][3].x()*term8;
  const double term34 = p[1][1].x()*term12 + p[1][2].x()*term13 + term3*term32 + term33*term7;
  const double term35 = term31*term34;
  const double term36 = term16 + term23 + term30 + term35;
  const double term37 = p[0][0].y()*term5;
  const double term38 = p[0][3].y()*term8;
  const double term39 = p[0][1].y()*term12 + p[0][2].y()*term13 + term3*term37 + term38*term7;
  const double term40 = term2*term39;
  const double term41 = term1*term40;
  const double term42 = p[3][0].y()*term5;
  const double term43 = p[3][3].y()*term8;
  const double term44 = p[3][1].y()*term12 + p[3][2].y()*term13 + term3*term42 + term43*term7;
  const double term45 = term18*term44;
  const double term46 = term17*term45;
  const double term47 = p[2][0].y()*term5;
  const double term48 = p[2][3].y()*term8;
  const double term49 = p[2][1].y()*term12 + p[2][2].y()*term13 + term3*term47 + term48*term7;
  const double term50 = term26*term49;
  const double term51 = p[1][0].y()*term5;
  const double term52 = p[1][3].y()*term8;
  const double term53 = p[1][1].y()*term12 + p[1][2].y()*term13 + term3*term51 + term52*term7;
  const double term54 = term31*term53;
  const double term55 = term41 + term46 + term50 + term54;
  const double term56 = p[0][0].z()*term5;
  const double term57 = p[0][3].z()*term8;
  const double term58 = p[0][1].z()*term12 + p[0][2].z()*term13 + term3*term56 + term57*term7;
  const double term59 = term2*term58;
  const double term60 = term1*term59;
  const double term61 = p[3][0].z()*term5;
  const double term62 = p[3][3].z()*term8;
  const double term63 = p[3][1].z()*term12 + p[3][2].z()*term13 + term3*term61 + term62*term7;
  const double term64 = term18*term63;
  const double term65 = term17*term64;
  const double term66 = p[2][0].z()*term5;
  const double term67 = p[2][3].z()*term8;
  const double term68 = p[2][1].z()*term12 + p[2][2].z()*term13 + term3*term66 + term67*term7;
  const double term69 = term26*term68;
  const double term70 = p[1][0].z()*term5;
  const double term71 = p[1][3].z()*term8;
  const double term72 = p[1][1].z()*term12 + p[1][2].z()*term13 + term3*term70 + term7*term71;
  const double term73 = term31*term72;
  const double term74 = term60 + term65 + term69 + term73;
  const double term75 = (term36 * term36) + (term55 * term55) + (term74 * term74);
  const double term76 = 1. / sqrt(term75);
  const double term77 = term1*term2;
  const double term78 = 0.166666666666667*term3;
  const double term79 = 0.166666666666667*term7;
  const double term80 = 1.5*term10;
  const double term81 = -11.0*frac_x + term80 + 19.5;
  const double term82 = 10*frac_x - term80 - 16;
  const double term83 = 2*frac_x;
  const double term84 = term83 - 8;
  const double term85 = term83 - 6;
  const double term86 = term17*term18;
  const double term87 = term26*(-p[2][0].x()*term78 + p[2][1].x()*term81 + p[2][2].x()*term82 + p[2][3].x()*term79 + term27*term84 + term28*term85) + term31*(-p[1][0].x()*term78 + p[1][1].x()*term81 + p[1][2].x()*term82 + p[1][3].x()*term79 + term32*term84 + term33*term85) + term77*(-p[0][0].x()*term78 + p[0][1].x()*term81 + p[0][2].x()*term82 + p[0][3].x()*term79 + term6*term84 + term85*term9) + term86*(-p[3][0].x()*term78 + p[3][1].x()*term81 + p[3][2].x()*term82 + p[3][3].x()*term79 + term19*term84 + term20*term85);
  const double term88b = 1. / sqrt(term75);
  const double term88 = term88b * term88b * term88b;
  const double term89 = (1.0L/2.0L)*term16 + (1.0L/2.0L)*term23 + (1.0L/2.0L)*term30 + (1.0L/2.0L)*term35;
  const double term90 = (1.0L/2.0L)*term41 + (1.0L/2.0L)*term46 + (1.0L/2.0L)*term50 + (1.0L/2.0L)*term54;
  const double term91 = term26*(-p[2][0].y()*term78 + p[2][1].y()*term81 + p[2][2].y()*term82 + p[2][3].y()*term79 + term47*term84 + term48*term85) + term31*(-p[1][0].y()*term78 + p[1][1].y()*term81 + p[1][2].y()*term82 + p[1][3].y()*term79 + term51*term84 + term52*term85) + term77*(-p[0][0].y()*term78 + p[0][1].y()*term81 + p[0][2].y()*term82 + p[0][3].y()*term79 + term37*term84 + term38*term85) + term86*(-p[3][0].y()*term78 + p[3][1].y()*term81 + p[3][2].y()*term82 + p[3][3].y()*term79 + term42*term84 + term43*term85);
  const double term92 = (1.0L/2.0L)*term60 + (1.0L/2.0L)*term65 + (1.0L/2.0L)*term69 + (1.0L/2.0L)*term73;
  const double term93 = term26*(-p[2][0].z()*term78 + p[2][1].z()*term81 + p[2][2].z()*term82 + p[2][3].z()*term79 + term66*term84 + term67*term85) + term31*(-p[1][0].z()*term78 + p[1][1].z()*term81 + p[1][2].z()*term82 + p[1][3].z()*term79 + term70*term84 + term71*term85) + term77*(-p[0][0].z()*term78 + p[0][1].z()*term81 + p[0][2].z()*term82 + p[0][3].z()*term79 + term56*term84 + term57*term85) + term86*(-p[3][0].z()*term78 + p[3][1].z()*term81 + p[3][2].z()*term82 + p[3][3].z()*term79 + term61*term84 + term62*term85);
  const double term94 = 2*term88*(term87*term89 + term90*term91 + term92*term93);
  const double term95 = 1.5*term24;
  const double term96 = 10*frac_y - term95 - 16;
  const double term97 = term29*term96;
  const double term98 = -11.0*frac_y + term95 + 19.5;
  const double term99 = term34*term98;
  const double term100 = 2*frac_y;
  const double term101 = term1*(term100 - 8);
  const double term102 = term101*term14;
  const double term103 = term17*(term100 - 6);
  const double term104 = term103*term21;
  const double term105 = term49*term96;
  const double term106 = term53*term98;
  const double term107 = term101*term39;
  const double term108 = term103*term44;
  const double term109 = term68*term96;
  const double term110 = term72*term98;
  const double term111 = term101*term58;
  const double term112 = term103*term63;
  const double term113 = term88*(term89*(2*term102 + 2*term104 - 0.333333333333333*term15 + 0.333333333333333*term22 + 2*term97 + 2*term99) + term90*(2*term105 + 2*term106 + 2*term107 + 2*term108 - 0.333333333333333*term40 + 0.333333333333333*term45) + term92*(2*term109 + 2*term110 + 2*term111 + 2*term112 - 0.333333333333333*term59 + 0.333333333333333*term64));
  
  (*result)[0] = term36*term76;
  (*result)[1] = term55*term76;
  (*result)[2] = term74*term76;

  (*dresult_dxy)(0, 0) = -term36*term94 + term76*term87;
  (*dresult_dxy)(0, 1) = -term113*term36 + term76*(term102 + term104 - 0.166666666666667*term15 + 0.166666666666667*term22 + term97 + term99);
  (*dresult_dxy)(1, 0) = -term55*term94 + term76*term91;
  (*dresult_dxy)(1, 1) = -term113*term55 + term76*(term105 + term106 + term107 + term108 - 0.166666666666667*term40 + 0.166666666666667*term45);
  (*dresult_dxy)(2, 0) = -term74*term94 + term76*term93;
  (*dresult_dxy)(2, 1) = -term113*term74 + term76*(term109 + term110 + term111 + term112 - 0.166666666666667*term59 + 0.166666666666667*term64);
}
}  // namespace aslam
