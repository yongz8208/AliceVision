
#pragma once

#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/camera/Pinhole.hpp>
#include <ceres/ceres.h>

namespace aliceVision {
namespace calibration {

class CostLine : public ceres::CostFunction
{
public:
    CostLine(std::shared_ptr<camera::Pinhole> & camera, const Vec2& pt)
        : _pt(pt)
        , _camera(camera)
    {
        set_num_residuals(1);

        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(2);
        mutable_parameter_block_sizes()->push_back(2);
        mutable_parameter_block_sizes()->push_back(camera->getDistortionParams().size());
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_angle_line = parameters[0];
        const double* parameter_dist_line = parameters[1];
        const double* parameter_scale = parameters[2];
        const double* parameter_center = parameters[3];
        const double* parameter_disto = parameters[4];

        const double angle = parameter_angle_line[0];
        const double distanceToLine = parameter_dist_line[0];

        const double cangle = cos(angle);
        const double sangle = sin(angle);

        const int distortionSize = _camera->getDistortionParams().size();

        //Read parameters and update camera
        _camera->setScale(parameter_scale[0], parameter_scale[1]);
        _camera->setOffset(parameter_center[0], parameter_center[1]);
        std::vector<double> cameraDistortionParams = _camera->getDistortionParams();

        for (int idParam = 0; idParam < distortionSize; idParam++)
        {
            cameraDistortionParams[idParam] = parameter_disto[idParam];
        }
        _camera->setDistortionParams(cameraDistortionParams);


        //Estimate measure
        const Vec2 cpt = _camera->ima2cam(_pt);
        const Vec2 distorted = _camera->addDistortion(cpt);
        const Vec2 ipt = _camera->cam2ima(distorted);

        const double w1 = std::max(0.4, std::max(std::abs(distorted.x()), std::abs(distorted.y())));
        const double w = w1 * w1;

        residuals[0] = w * (cangle * ipt.x() + sangle * ipt.y() - distanceToLine);

        if(jacobians == nullptr)
        {
            return true;
        }

        if(jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[0]);

            J(0, 0) = w * (ipt.x() * -sangle + ipt.y() * cangle);
        }

        if(jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> J(jacobians[1]);
            J(0, 0) = -w;
        }

        if(jacobians[2] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>> J(jacobians[2]);
            
            Eigen::Matrix<double, 1, 2> Jline;
            Jline(0, 0) = cangle;
            Jline(0, 1) = sangle;

            J = w * Jline * (_camera->getDerivativeIma2CamWrtScale(distorted) + _camera->getDerivativeCam2ImaWrtPoint() * _camera->getDerivativeAddDistoWrtPt(cpt) * _camera->getDerivativeIma2CamWrtScale(_pt));
        }

        if(jacobians[3] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>> J(jacobians[3]);

            Eigen::Matrix<double, 1, 2> Jline;
            Jline(0, 0) = cangle;
            Jline(0, 1) = sangle;

            J = w * Jline * (_camera->getDerivativeCam2ImaWrtPrincipalPoint() + _camera->getDerivativeCam2ImaWrtPoint() * _camera->getDerivativeAddDistoWrtPt(cpt) * _camera->getDerivativeIma2CamWrtPrincipalPoint());
        }

        if(jacobians[4] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[4], 1, distortionSize);

            Eigen::Matrix<double, 1, 2> Jline;
            Jline(0, 0) = cangle;
            Jline(0, 1) = sangle;

            J = w * Jline * _camera->getDerivativeCam2ImaWrtPoint() * _camera->getDerivativeAddDistoWrtDisto(cpt);
        }

        return true;
    }

private:
    std::shared_ptr<camera::Pinhole> _camera;
    Vec2 _pt;
};


class CostPoint : public ceres::CostFunction
{
public:
    CostPoint(std::shared_ptr<camera::Pinhole> & camera, const Vec2& ptUndistorted, const Vec2 &ptDistorted)
        : _ptUndistorted(ptUndistorted)
        , _ptDistorted(ptDistorted)
        , _camera(camera)
    {
        set_num_residuals(2);

        mutable_parameter_block_sizes()->push_back(2);
        mutable_parameter_block_sizes()->push_back(2);
        mutable_parameter_block_sizes()->push_back(camera->getDistortionParams().size());
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const double* parameter_scale = parameters[0];
        const double* parameter_center = parameters[1];
        const double* parameter_disto = parameters[2];

        const int distortionSize = _camera->getDistortionParams().size();

        //Read parameters and update camera
        _camera->setScale(parameter_scale[0], parameter_scale[1]);
        _camera->setOffset(parameter_center[0], parameter_center[1]);
        std::vector<double> cameraDistortionParams = _camera->getDistortionParams();

        for (int idParam = 0; idParam < distortionSize; idParam++)
        {
            cameraDistortionParams[idParam] = parameter_disto[idParam];
        }
        _camera->setDistortionParams(cameraDistortionParams);

        //Estimate measure
        const Vec2 cpt = _camera->ima2cam(_ptUndistorted);
        const Vec2 distorted = _camera->addDistortion(cpt);
        const Vec2 ipt = _camera->cam2ima(distorted);

        const double w1 = std::max(std::abs(distorted.x()), std::abs(distorted.y()));
        const double w = w1 * w1;

        residuals[0] = w * (ipt.x() - _ptDistorted.x());
        residuals[1] = w * (ipt.y() - _ptDistorted.y());

        if(jacobians == nullptr)
        {
            return true;
        }

        if(jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J(jacobians[0]);
            
            J = w * (_camera->getDerivativeIma2CamWrtScale(distorted) + _camera->getDerivativeCam2ImaWrtPoint() * _camera->getDerivativeAddDistoWrtPt(cpt) * _camera->getDerivativeIma2CamWrtScale(_ptUndistorted));
        }

        if(jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J(jacobians[1]);


            J = w * (_camera->getDerivativeCam2ImaWrtPrincipalPoint() + _camera->getDerivativeCam2ImaWrtPoint() * _camera->getDerivativeAddDistoWrtPt(cpt) * _camera->getDerivativeIma2CamWrtPrincipalPoint());
        }

        if(jacobians[2] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[2], 2, distortionSize);

            J = w * _camera->getDerivativeCam2ImaWrtPoint() * _camera->getDerivativeAddDistoWrtDisto(cpt);
        }

        return true;
    }

private:
    std::shared_ptr<camera::Pinhole> _camera;
    Vec2 _ptUndistorted;
    Vec2 _ptDistorted;
};

}//namespace calibration
}//namespace aliceVision
