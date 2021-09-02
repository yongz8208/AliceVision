// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2021 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/sfmData/SfMData.hpp>

#include <aliceVision/sfm/pipeline/ReconstructionEngine.hpp>
#include <aliceVision/feature/FeaturesPerView.hpp>
#include <aliceVision/matching/IndMatch.hpp>
#include <aliceVision/multiview/rotationAveraging/rotationAveraging.hpp>

namespace aliceVision{
namespace sfm{


/**
 * Global rotation Pipeline Reconstruction Engine.
 * The method is based on the Global SfM but with no translations between cameras.
 */
class ReconstructionEngine_globalRotation : public ReconstructionEngine
{
public:

  struct Params
  {
    bool rotationAveragingWeighting = true;
  };

  ReconstructionEngine_globalRotation(const sfmData::SfMData& sfmData, const Params& params, const std::string& outDirectory);
  virtual ~ReconstructionEngine_globalRotation();

  void SetFeaturesProvider(feature::FeaturesPerView* featuresPerView);
  void SetMatchesProvider(matching::PairwiseMatches* provider);

  virtual bool process();
  void filterMatches();

private:
  void computeRelativeRotations(aliceVision::rotationAveraging::RelativeRotations& vec_relatives_R);
  void buildRotationPriors();

private:
  // Parameter
  Params _params;

  // Data provider
  feature::FeaturesPerView* _featuresPerView;
  matching::PairwiseMatches* _pairwiseMatches;
};

} // namespace sfm
} // namespace aliceVision
