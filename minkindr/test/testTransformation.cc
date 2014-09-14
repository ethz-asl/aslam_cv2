#include <gtest/gtest.h>
#include <kindr/minimal/quat-transformation.h>
#include <cmath>

typedef kindr::minimal::QuatTransformation Transformation;

Eigen::Vector3d fromHomogeneous(const Eigen::Vector4d& v) {
  return v.head<3>() / v[3];
}


TEST(MinKindrTests, testTransform) {
  using namespace kindr::minimal;
  Eigen::Vector4d q(0.64491714, 0.26382416,  0.51605132,  0.49816637);
  RotationQuaternion q1(q);
  Eigen::Vector3d t( 4.67833851,  8.52053031,  6.71796159 );

  Transformation T(q,t);
  //std::cout << "T:\n" << T << std::endl;
  Transformation invT = T.inverted();
  //std::cout << "invT:\n" << invT << std::endl;
  Eigen::Vector3d v(6.26257419,  1.58356548,  6.05772983);
  Eigen::Vector4d vh(6.26257419,  1.58356548,  6.05772983, 1.0);
  Eigen::Vector4d vh2 = vh * 0.2;
  Eigen::Vector4d vh3 = -vh * 0.2;
  
  Eigen::Vector3d Tv( 9.53512701,  15.88020996,   7.53669644);
  Eigen::Vector4d Tvh(9.53512701,  15.88020996,   7.53669644, 1.0);
  Eigen::Vector3d invTv(-6.12620997, -3.67891623,  0.04812912);
  Eigen::Vector3d Cv(4.8567885 ,  7.35967965,  0.81873485);
  Eigen::Vector3d invCv(-1.17246549,  3.4343828 ,  8.07799141);

  Eigen::Matrix4d Tmx;
  Tmx << 0.3281757 , -0.17386938,  0.92847733,  2.7527055,
         0.85444827, -0.36445416, -0.37025845,  0.7790679,
         0.40276403,  0.91484567,  0.02895739,  4.1696795,
      0.        ,  0.        ,  0.        ,  1. ;
  
  
  Eigen::Vector3d Tv1 = T.transform(v);
  Eigen::Vector4d Tvh2 = T.transform4(vh);
  Eigen::Vector4d Tvh3 = T.transform4(vh2);
  Eigen::Vector4d Tvh4 = T.transform4(vh3);
  Eigen::Vector3d Tv2 = fromHomogeneous(Tvh2);
  Eigen::Vector3d Tv3 = fromHomogeneous(Tvh3);
  Eigen::Vector3d Tv4 = fromHomogeneous(Tvh4);

  for(int i = 0; i < 3; ++i) {
    EXPECT_NEAR(Tv1[i], Tv[i], 1e-4);
    EXPECT_NEAR(Tv2[i], Tv[i], 1e-4);
    EXPECT_NEAR(Tv3[i], Tv[i], 1e-4);
    EXPECT_NEAR(Tv4[i], Tv[i], 1e-4);
  }

  {
    Eigen::Vector3d invTv1 = T.inverted().transform(v);
    Eigen::Vector4d invTvh2 = T.inverted().transform4(vh);
    Eigen::Vector4d invTvh3 = T.inverted().transform4(vh2);
    Eigen::Vector4d invTvh4 = T.inverted().transform4(vh3);
    Eigen::Vector3d invTv2 = fromHomogeneous(invTvh2);
    Eigen::Vector3d invTv3 = fromHomogeneous(invTvh3);
    Eigen::Vector3d invTv4 = fromHomogeneous(invTvh4);
  
    for(int i = 0; i < 3; ++i) {
      EXPECT_NEAR(invTv1[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv2[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv3[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv4[i], invTv[i], 1e-4);
    }
  }

  {
    Eigen::Vector3d invTv1 = invT.transform(v);
    Eigen::Vector4d invTvh2 = invT.transform4(vh);
    Eigen::Vector4d invTvh3 = invT.transform4(vh2);
    Eigen::Vector4d invTvh4 = invT.transform4(vh3);
    Eigen::Vector3d invTv2 = fromHomogeneous(invTvh2);
    Eigen::Vector3d invTv3 = fromHomogeneous(invTvh3);
    Eigen::Vector3d invTv4 = fromHomogeneous(invTvh4);
  
    for(int i = 0; i < 3; ++i) {
      EXPECT_NEAR(invTv1[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv2[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv3[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv4[i], invTv[i], 1e-4);
    }
  }

  {
    Eigen::Vector3d invTv1 = T.inverseTransform(v);
    Eigen::Vector4d invTvh2 = T.inverseTransform4(vh);
    Eigen::Vector4d invTvh3 = T.inverseTransform4(vh2);
    Eigen::Vector4d invTvh4 = T.inverseTransform4(vh3);
    Eigen::Vector3d invTv2 = fromHomogeneous(invTvh2);
    Eigen::Vector3d invTv3 = fromHomogeneous(invTvh3);
    Eigen::Vector3d invTv4 = fromHomogeneous(invTvh4);
  
    for(int i = 0; i < 3; ++i) {
      EXPECT_NEAR(invTv1[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv2[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv3[i], invTv[i], 1e-4);
      EXPECT_NEAR(invTv4[i], invTv[i], 1e-4);
    }
  }

  {
    Eigen::Vector3d invCv1 = T.inverseTransformVector(v);
    Eigen::Vector3d invCv2 = T.inverted().transformVector(v);
  
    for(int i = 0; i < 3; ++i) {
      EXPECT_NEAR(invCv1[i], invCv[i], 1e-4);
      EXPECT_NEAR(invCv2[i], invCv[i], 1e-4);
    }
  }

  {
    Eigen::Vector3d Cv1 = T.transformVector(v);
  
    for(int i = 0; i < 3; ++i) {
      EXPECT_NEAR(Cv1[i], Cv[i], 1e-4);
    }
  }


}

TEST(MinKindrTests, testCompose) {
  using namespace kindr::minimal;
  Eigen::Vector4d q(0.64491714, 0.26382416,  0.51605132,  0.49816637);
  RotationQuaternion q1(q);
  Eigen::Vector3d t( 4.67833851,  8.52053031,  6.71796159 );

  Transformation T(q,t);
  //std::cout << "T:\n" << T << std::endl;
  Transformation invT = T.inverted();
  //std::cout << "invT:\n" << invT << std::endl;
  Eigen::Vector3d v(6.26257419,  1.58356548,  6.05772983);
  Eigen::Vector3d invTTv(-8.16137069,  -6.14469052, -14.34176544);
  Eigen::Vector3d TTv(5.52009598,  24.34170933,  18.9197339);

  
  Transformation TT = T*T;
  Transformation Id1 = T*T.inverted();
  Transformation Id2 = T.inverted() * T;
  Transformation iTiT1 = T.inverted() * T.inverted();
  Transformation iTiT2 = TT.inverted();

  

  Eigen::Vector3d TTv1 = TT.transform(v);
  for(int i = 0; i < 3; ++i) {
    EXPECT_NEAR(TTv1[i], TTv[i],1e-4);
  }

  Eigen::Vector3d v1 = Id1.transform(v);
  Eigen::Vector3d v2 = Id2.transform(v);
  for(int i = 0; i < 3; ++i) {
    EXPECT_NEAR(v1[i], v[i],1e-4);
    EXPECT_NEAR(v2[i], v[i],1e-4);
  }

  Eigen::Vector3d iTTv1 = iTiT1.transform(v);
  Eigen::Vector3d iTTv2 = iTiT2.transform(v);
  for(int i = 0; i < 3; ++i) {
    EXPECT_NEAR(iTTv1[i], invTTv[i],1e-4);
    EXPECT_NEAR(iTTv2[i], invTTv[i],1e-4);
  }

  
}

