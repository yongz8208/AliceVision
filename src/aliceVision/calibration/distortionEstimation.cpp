// This file is part of the AliceVision project.
// Copyright (c) 2021 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "distortionEstimation.hpp"

#include <ceres/ceres.h>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/calibration/distortionCosts.hpp>


namespace aliceVision {
namespace calibration {

bool estimate(std::shared_ptr<camera::Pinhole> & cameraToEstimate, Statistics & statistics, std::vector<LineWithPoints> & lines, bool lockScale, bool lockCenter, const std::vector<bool> & lockDistortions)
{
    if (!cameraToEstimate)
    {
        return false; 
    }

    if (lines.empty())
    {
        return false;
    }

    const size_t countDistortionParams = cameraToEstimate->getDistortionParams().size();
    if (lockDistortions.size() != countDistortionParams) 
    {
        ALICEVISION_LOG_ERROR("Invalid number of distortion parameters (lockDistortions=" << lockDistortions.size() << ", countDistortionParams=" << countDistortionParams << ").");
        return false;
    }

    ceres::Problem problem;
    ceres::LossFunction* lossFunction = nullptr;

    std::vector<double> params = cameraToEstimate->getParams();
    double * scale = &params[0];
    double * center = &params[2];
    double * distortionParameters = &params[4];

    problem.AddParameterBlock(scale, 2);
    if (lockScale)
    {
        problem.SetParameterBlockConstant(scale);
    }
    else
    {
        ceres::SubsetParameterization* subsetParameterization = new ceres::SubsetParameterization(2, {1});   
        problem.SetParameterization(scale, subsetParameterization);
    }

    //Add off center parameter
    problem.AddParameterBlock(center, 2);
    if (lockCenter)
    {
        problem.SetParameterBlockConstant(center);
    }

    //Add distortion parameter
    problem.AddParameterBlock(distortionParameters, countDistortionParams);

    //Check if all distortions are locked 
    bool allLocked = true;
    for (bool lock : lockDistortions) 
    {
        if (!lock)
        {
            allLocked = false;
        }
    }

    if (allLocked)
    {
        problem.SetParameterBlockConstant(distortionParameters);
    }
    else 
    {
        //At least one parameter is not locked

        std::vector<int> constantDistortions;
        for (int idParamDistortion = 0; idParamDistortion < lockDistortions.size(); idParamDistortion++)
        {
            if (lockDistortions[idParamDistortion])
            {
                constantDistortions.push_back(idParamDistortion);
            }
        }

        if (!constantDistortions.empty())
        {
            ceres::SubsetParameterization* subsetParameterization = new ceres::SubsetParameterization(countDistortionParams, constantDistortions);   
            problem.SetParameterization(distortionParameters, subsetParameterization);
        }
    }
    
    
    for (auto & l : lines)
    {
        problem.AddParameterBlock(&l.angle, 1);
        problem.AddParameterBlock(&l.dist, 1);

        for (Vec2 pt : l.points)
        {
            ceres::CostFunction * costFunction = new CostLine(cameraToEstimate, pt);   
            problem.AddResidualBlock(costFunction, lossFunction, &l.angle, &l.dist, scale, center, distortionParameters);
        }
    }

    ceres::Solver::Options options;
    options.use_inner_iterations = true;
    options.max_num_iterations = 10000; 
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;  
    ceres::Solve(options, &problem, &summary);

    ALICEVISION_LOG_TRACE(summary.FullReport());

    if (!summary.IsSolutionUsable())
    {
        ALICEVISION_LOG_ERROR("Lens calibration estimation failed.");
        return false;
    }

    cameraToEstimate->updateFromParams(params);

    std::vector<double> errors;

    for (auto & l : lines)
    {
        const double sangle = sin(l.angle);
        const double cangle = cos(l.angle);

        for(const Vec2& pt : l.points)
        {
            const Vec2 cpt = cameraToEstimate->ima2cam(pt);
            const Vec2 distorted = cameraToEstimate->addDistortion(cpt);
            const Vec2 ipt = cameraToEstimate->cam2ima(distorted);

            const double res = (cangle * ipt.x() + sangle * ipt.y() - l.dist);

            errors.push_back(std::abs(res));
        }
    }

    const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / double(errors.size());
    const double sqSum = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
    const double stddev = std::sqrt(sqSum / errors.size() - mean * mean);
    std::nth_element(errors.begin(), errors.begin() + errors.size()/2, errors.end());
    const double median = errors[errors.size() / 2];

    statistics.mean = mean;
    statistics.stddev = stddev;
    statistics.median = median;

    return true;
}

bool estimate(std::shared_ptr<camera::Pinhole> & cameraToEstimate, Statistics & statistics, std::vector<PointPair> & points, bool lockScale, bool lockCenter, const std::vector<bool> & lockDistortions)
{
    if (!cameraToEstimate)
    {
        return false; 
    }

    if (points.empty())
    {
        return false;
    }

    size_t countDistortionParams = cameraToEstimate->getDistortionParams().size();
    if (lockDistortions.size() != countDistortionParams) 
    {
        ALICEVISION_LOG_ERROR("Invalid number of distortion parameters (lockDistortions=" << lockDistortions.size() << ", countDistortionParams=" << countDistortionParams << ").");
        return false;
    }

    ceres::Problem problem;
    ceres::LossFunction* lossFunction = nullptr;

    std::vector<double> params = cameraToEstimate->getParams();
    double * scale = &params[0];
    double * center = &params[2];
    double * distortionParameters = &params[4];

    problem.AddParameterBlock(scale, 2);
    if (lockScale)
    {
        problem.SetParameterBlockConstant(scale);
    }
    else
    {
        ceres::SubsetParameterization* subsetParameterization = new ceres::SubsetParameterization(2, {1});   
        problem.SetParameterization(scale, subsetParameterization);
    }
    

    //Add off center parameter
    problem.AddParameterBlock(center, 2);
    if (lockCenter)
    {
        problem.SetParameterBlockConstant(center);
    }

    //Add distortion parameter
    problem.AddParameterBlock(distortionParameters, countDistortionParams);

    //Check if all distortions are locked 
    bool allLocked = true;
    for (bool lock : lockDistortions) 
    {
        if (!lock)
        {
            allLocked = false;
        }
    }

    if (allLocked)
    {
        problem.SetParameterBlockConstant(distortionParameters);
    }
    else 
    {
        //At least one parameter is not locked

        std::vector<int> constantDistortions;
        for (int idParamDistortion = 0; idParamDistortion < lockDistortions.size(); idParamDistortion++)
        {
            if (lockDistortions[idParamDistortion])
            {
                constantDistortions.push_back(idParamDistortion);
            }
        }
        
        if (!constantDistortions.empty())
        {
            ceres::SubsetParameterization* subsetParameterization = new ceres::SubsetParameterization(countDistortionParams, constantDistortions);   
            problem.SetParameterization(distortionParameters, subsetParameterization);
        }
    }
    
    for (const PointPair & pt : points)
    {
        ceres::CostFunction * costFunction = new CostPoint(cameraToEstimate, pt.undistortedPoint, pt.distortedPoint);   
        problem.AddResidualBlock(costFunction, lossFunction, scale, center, distortionParameters);
    }

    // google::SetCommandLineOption("GLOG_minloglevel", "3");
    ceres::Solver::Options options;
    options.use_inner_iterations = true;
    options.max_num_iterations = 10000; 
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;  
    ceres::Solve(options, &problem, &summary);

    ALICEVISION_LOG_TRACE(summary.FullReport());

    if (!summary.IsSolutionUsable())
    {
        ALICEVISION_LOG_ERROR("Lens calibration estimation failed.");
        return false;
    }

    cameraToEstimate->updateFromParams(params);

    std::vector<double> errors;

    for (PointPair pp : points)
    {
        const Vec2 cpt = cameraToEstimate->ima2cam(pp.undistortedPoint);
        const Vec2 distorted = cameraToEstimate->addDistortion(cpt);
        const Vec2 ipt = cameraToEstimate->cam2ima(distorted);

        const double res = (ipt - pp.distortedPoint).norm();

        errors.push_back(res);
    }

    const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / double(errors.size());
    const double sqSum = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
    const double stddev = std::sqrt(sqSum / errors.size() - mean * mean);
    std::nth_element(errors.begin(), errors.begin() + errors.size()/2, errors.end());
    const double median = errors[errors.size() / 2];

    statistics.mean = mean;
    statistics.stddev = stddev;
    statistics.median = median;

    return true;
}

}//namespace calibration
}//namespace aliceVision
