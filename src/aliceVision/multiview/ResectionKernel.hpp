// This file is part of the AliceVision project.
// Copyright (c) 2019 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/config.hpp>
#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/robustEstimation/conditioning.hpp>
#include <aliceVision/robustEstimation/ISolver.hpp>
#include <aliceVision/robustEstimation/PointFittingRansacKernel.hpp>

namespace aliceVision {
namespace multiview {

/**
 * @brief Pose/Resection Kernel with unknown intrinsic for the A contrario (AC) model estimator.
 *        to be used with the ACRANSAC / LORANSAC framework.
 *
 * @ref [1] "Robust and accurate calibration of camera networks". PhD. Pierre MOULON
 */
template<typename SolverT_, typename ErrorT_, typename UnnormalizerT_, typename ModelT_ = robustEstimation::Mat34Model, typename SolverLsT_ = robustEstimation::UndefinedSolver<ModelT_>>
class ResectionKernel
    : public robustEstimation::PointFittingRansacKernel<SolverT_, ErrorT_, ModelT_, SolverLsT_>
{
public:

  using KernelBase = robustEstimation::PointFittingRansacKernel<SolverT_, ErrorT_, ModelT_, SolverLsT_>;

  /**
   * @brief ResectionKernel constructor with unknown intrinsic
   * @param[in] x2d 2d points in the first image. One per column.
   * @param[in] x3d 3d points in the second image. One per column.
   * @param[in] K intrinsic camera parameter
   */
  ResectionKernel(const Mat& x2d, const Mat& x3d, int w, int h)
    : _x2d(x2d.rows(), x2d.cols())
    , KernelBase(_x2d, x3d)
    , _logalpha0(log10(M_PI))
    , _N1(3, 3)
  {
    assert(2 == x2d.rows());
    assert(3 == x3d.rows());
    assert(x2d.cols() == x3d.cols());

    robustEstimation::normalizePointsFromImageSize(x2d, &_x2d, &_N1, w, h);
  }

  void unnormalize(ModelT_& model) const override
  {
    // unnormalize model from the computed conditioning.
    Mat34 P = model.getMatrix();
    UnnormalizerT_::unnormalize(_N1, Mat3::Identity(), &P);
    model.setMatrix(P);
  }

  double logalpha0() const override {return _logalpha0; }
  double multError() const override {return 1.0;} // point to point error
  Mat3 normalizer1() const override {return Mat3::Identity();}
  Mat3 normalizer2() const override {return _N1;}
  double unormalizeError(double val) const override {return sqrt(val) / _N1(0,0);}

protected:
  /// Normalized input data
  Mat _x2d;
  /// Matrix used to normalize data
  Mat3 _N1;
  /// Alpha0 is used to make the error adaptive to the image size
  double _logalpha0;
};


/**
 * @brief The kernel for the resection with known intrinsics (PnP) to be used with
 * the ACRANSAC / LORANSAC framework.
 *
 * @ref [1] "Robust and accurate calibration of camera networks". PhD. Pierre MOULON
 *
 * @tparam SolverT The minimal solver able to find a solution from a minimum set of points, usually any PnP solver.
 * @tparam ErrorT The functor computing the error for each data sample with respect to the estimated model, usually a reprojection error functor.
 * @tparam UnnormalizerT The functor used to normalize the data before the estimation of the model, usually a functor that normalize the point in camera
 *         coordinates (ie multiply by the inverse of the calibration matrix).
 * @tparam SolverLsT = SolverT The least square solver that is used to find a solution from any set of data larger than the minimum required,
 *          usually the 6 point algorithm which solves the resection problem by means of LS.
 * @tparam ModelT = Mat34Model The type of the model to estimate, the projection matrix.
 */
template<typename SolverT_, typename ErrorT_, typename UnnormalizerT_, typename ModelT_ = robustEstimation::Mat34Model, typename SolverLsT_ = robustEstimation::UndefinedSolver<ModelT_>>
class ResectionKernel_K
    : public robustEstimation::PointFittingRansacKernel<SolverT_, ErrorT_, ModelT_, SolverLsT_>
{
public:

  using KernelBase = robustEstimation::PointFittingRansacKernel<SolverT_, ErrorT_, ModelT_, SolverLsT_>;

  /**
   * @brief ResectionKernel constructor with known intrinsic
   * @param[in] x2d 2d points in the first image. One per column.
   * @param[in] x3d 3d points in the second image. One per column.
   * @param[in] K intrinsic camera parameter
   */
  ResectionKernel_K(const Mat& x2d, const Mat& x3d, const Mat3& K)
    : _x2d(x2d.rows(), x2d.cols())
    , KernelBase(_x2d, x3d)
    , _logalpha0(log10(M_PI))
    , _N1(K.inverse())
    , _K(K)
  {
    assert(2 == x2d.rows());
    assert(3 == x3d.rows());
    assert(x2d.cols() == x3d.cols());

    // normalize points by inverse K
    robustEstimation::applyTransformationToPoints(x2d, _N1, &_x2d);
  }

  void unnormalize(ModelT_& model) const override
  {
    // unnormalize model from the computed conditioning.
    model.setMatrix(_K * model.getMatrix());
  }

  double logalpha0() const override {return _logalpha0; }
  double multError() const override {return 1.0;} // point to point error
  Mat3 normalizer1() const override {return Mat3::Identity();}
  Mat3 normalizer2() const override {return _N1;}
  double unormalizeError(double val) const override {return sqrt(val) / _N1(0,0);}

protected:
  /// Normalized input data
  Mat _x2d;
  /// Matrix used to normalize data
  Mat3 _N1;
  /// Alpha0 is used to make the error adaptive to the image size
  double _logalpha0;
  /// intrinsic camera parameter
  Mat3 _K;
};

/**
 * @brief The kernel for the resection with known intrinsics (PnP) and previous normals
 *
 * @tparam SolverT The minimal solver able to find a solution from a minimum set of points, usually any PnP solver.
 * @tparam ErrorT The functor computing the error for each data sample with respect to the estimated model, usually a reprojection error functor.
 * @tparam UnnormalizerT The functor used to normalize the data before the estimation of the model, usually a functor that normalize the point in camera
 *         coordinates (ie multiply by the inverse of the calibration matrix).
 * @tparam SolverLsT = SolverT The least square solver that is used to find a solution from any set of data larger than the minimum required,
 *          usually the 6 point algorithm which solves the resection problem by means of LS.
 * @tparam ModelT = Mat34Model The type of the model to estimate, the projection matrix.
 */
template<typename SolverT_, typename ErrorT_, typename UnnormalizerT_, typename ModelT_ = robustEstimation::Mat34Model, typename SolverLsT_ = robustEstimation::UndefinedSolver<ModelT_>>
class ResectionKernel_K_normals
    : public robustEstimation::PointFittingRansacKernel<SolverT_, ErrorT_, ModelT_, SolverLsT_>
{
public:

  using KernelBase = robustEstimation::PointFittingRansacKernel<SolverT_, ErrorT_, ModelT_, SolverLsT_>;

  /**
   * @brief ResectionKernel constructor with known intrinsic
   * @param[in] x2d 2d points in the first image. One per column.
   * @param[in] x3d 3d points in the second image. One per column.
   * @param[in] K intrinsic camera parameter
   */
  ResectionKernel_K_normals(const Mat& x2d, const Mat& x3d, const std::vector<std::vector<Vec3>> & normals, const Mat3& K)
    : _x2d(x2d.rows(), x2d.cols())
    , KernelBase(_x2d, x3d)
    , _logalpha0(log10(M_PI))
    , _N1(K.inverse())
    , _K(K)
    , _normals(normals)
    , _x3d(x3d)
  {
    assert(2 == x2d.rows());
    assert(3 == x3d.rows());
    assert(x2d.cols() == x3d.cols());

    // normalize points by inverse K
    robustEstimation::applyTransformationToPoints(x2d, _N1, &_x2d);
  }

  void unnormalize(ModelT_& model) const override
  {
    // unnormalize model from the computed conditioning.
    model.setMatrix(_K * model.getMatrix());
  }

  /**
   * @brief Function that computes the estimation error for a given model and all the elements.
   * @param[in] model The model to consider.
   * @param[out] vec_errors The vector containing all the estimation errors for every element.
   */
  void errors(const ModelT_& model, std::vector<double>& errors) const override
  {
    KernelBase::errors(model, errors);

    Mat34 P = model.getMatrix();
    Mat3 R = P.block<3,3>(0, 0);
    Vec3 t = P.block<3,1>(0, 3);
    Vec3 c = -R*t;

    for(std::size_t sample = 0; sample < _x3d.cols(); ++sample)
    {
      Vec3 pt3d = _x3d.col(sample);
      Vec3 dir = (c - pt3d).normalized();


      double minAngle = std::numeric_limits<double>::max();
      for (size_t index = 0; index < _normals[sample].size(); index++)
      {
        Vec3 otherDir = _normals[sample][index];

        double angle = camera::angleBetweenRays(dir, otherDir);
        minAngle = std::min(angle, minAngle);
      }

      if (_normals[sample].size() > 0)
      {
        if (minAngle > 40)
        {
          //errors[sample] = std::numeric_limits<double>::max();
        }
      }
    }
  }


  double logalpha0() const override {return _logalpha0; }
  double multError() const override {return 1.0;} // point to point error
  Mat3 normalizer1() const override {return Mat3::Identity();}
  Mat3 normalizer2() const override {return _N1;}
  double unormalizeError(double val) const override {return sqrt(val) / _N1(0,0);}

protected:
  /// 3d points
  Mat _x3d;
  /// Normalized input data
  Mat _x2d;
  /// Matrix used to normalize data
  Mat3 _N1;
  /// Alpha0 is used to make the error adaptive to the image size
  double _logalpha0;
  /// intrinsic camera parameter
  Mat3 _K;
  ///Normals
  std::vector<std::vector<Vec3>> _normals;
};


} // namespace multiview
} // namespace aliceVision
