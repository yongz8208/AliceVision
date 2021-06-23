#pragma once

#include <aliceVision/calibration/distortionEstimation.hpp>

namespace aliceVision {
namespace sfmData {

struct DistortionPattern
{
    IndexT intrinsicId;
    std::vector<calibration::PointPair> pointPairs;
};

}
}