// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2021 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ReconstructionEngine_globalRotation.hpp"

#include <aliceVision/sfm/filters.hpp>
#include <aliceVision/graph/connectedComponent.hpp>
#include <aliceVision/sfmData/CameraPose.hpp>

namespace aliceVision {
namespace sfm {

typedef struct TRiggedPose
{
  IndexT poseId;
  IndexT rigId;
  IndexT subPoseId;

  bool operator==(const TRiggedPose & other) const
  {
    return (poseId == other.poseId && rigId == other.rigId && subPoseId == other.subPoseId);
  }

  bool operator<(const TRiggedPose & other) const
  {
    if (poseId < other.poseId)
    {
      return true;
    }
    else if( poseId > other.poseId)
    {
      return false;
    }
    
    if (rigId < other.rigId)
    {
      return true;
    }
    else if (rigId > other.rigId)
    {
      return false;
    }

    if (subPoseId < other.subPoseId)
    {
      return true;
    }
    else if (subPoseId > other.subPoseId)
    {
      return false;
    }

    return false;
  }
} RiggedPose;

ReconstructionEngine_globalRotation::ReconstructionEngine_globalRotation(const sfmData::SfMData& sfmData, const ReconstructionEngine_globalRotation::Params& params, const std::string& outDirectory)
  : ReconstructionEngine(sfmData, outDirectory)
  , _params(params)
{
}

ReconstructionEngine_globalRotation::~ReconstructionEngine_globalRotation()
{
}

void ReconstructionEngine_globalRotation::SetFeaturesProvider(feature::FeaturesPerView* featuresPerView)
{
  _featuresPerView = featuresPerView;
}

void ReconstructionEngine_globalRotation::SetMatchesProvider(matching::PairwiseMatches* provider)
{
  _pairwiseMatches = provider;
}

bool ReconstructionEngine_globalRotation::process()
{
  buildRotationPriors();

  aliceVision::rotationAveraging::RelativeRotations relatives_R;
  computeRelativeRotations(relatives_R);

  return true;
}

void ReconstructionEngine_globalRotation::filterMatches()
{
    // keep only the largest biedge connected subgraph
    const PairSet pairs = matching::getImagePairs(*_pairwiseMatches);
    const std::set<IndexT> set_remainingIds = graph::CleanGraph_KeepLargestBiEdge_Nodes<PairSet, IndexT>(pairs, _outputFolder);
    
    KeepOnlyReferencedElement(set_remainingIds, *_pairwiseMatches);
}

void ReconstructionEngine_globalRotation::buildRotationPriors()
{
  sfmData::RotationPriors & rotationpriors = _sfmData.getRotationPriors();
  rotationpriors.clear();

  for (auto & iter_v1 :_sfmData.getViews()) {

    if (!_sfmData.isPoseAndIntrinsicDefined(iter_v1.first)) {
      continue;
    }

    for (auto & iter_v2 :_sfmData.getViews()) {
      if (iter_v1.first == iter_v2.first) {
        continue;
      }

      if (!_sfmData.isPoseAndIntrinsicDefined(iter_v2.first)) {
        continue;
      }

      IndexT pid1 = iter_v1.second->getPoseId();
      IndexT pid2 = iter_v2.second->getPoseId();

      sfmData::CameraPose oneTo = _sfmData.getAbsolutePose(iter_v1.second->getPoseId());
      sfmData::CameraPose twoTo = _sfmData.getAbsolutePose(iter_v2.second->getPoseId());
      Eigen::Matrix3d oneRo = oneTo.getTransform().rotation();
      Eigen::Matrix3d twoRo = twoTo.getTransform().rotation();
      Eigen::Matrix3d twoRone = twoRo * oneRo.transpose();

      sfmData::RotationPrior prior(iter_v1.first, iter_v2.first, twoRone); 
      rotationpriors.push_back(prior);
    }
  }
}

void ReconstructionEngine_globalRotation::computeRelativeRotations(aliceVision::rotationAveraging::RelativeRotations& vec_relatives_R)
{
  vec_relatives_R.clear();

  const sfmData::RotationPriors & rotationpriors = _sfmData.getRotationPriors();
  for (auto it : rotationpriors)
  {
    vec_relatives_R.emplace_back(it.ViewFirst, it.ViewSecond, it._second_R_first, _params.rotationAveragingWeighting ? 1.0 : 0.01);
  }
}


} // namespace sfm
} // namespace aliceVision

