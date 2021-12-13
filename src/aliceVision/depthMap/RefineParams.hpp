// This file is part of the AliceVision project.
// Copyright (c) 2021 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

namespace aliceVision {
namespace depthMap {

/**
 * @brief Refine Parameters
 */
struct RefineParams
{
  // user parameters

  int wsh = 3;
  int maxTCams = 6;
  int nDepthsToRefine = 31;
  int nSamplesHalf = 150;
  int nIters = 100;
  int smoothXYZVolumeRadius = -1; // if <= 0, do nothing
  int smoothZVolumeRadius = -1;   // if <= 0, do nothing
  double sigma = 15.0;
  double gammaC = 15.5;
  double gammaP = 8.0;
  bool useTcOrRcPixSize = false;
  bool doRefineFuse = true;
  bool doRefineFuseLegacy = false;
  bool doVolumeOptimization = true;
  bool doDepthSimMapOptimization = true;
  bool interpolateRetrieveBestDepth = true; // only for refineFuseVolume (not legacy)
  bool exportIntermediateResults = false;

  // constant parameters

  const int scale = 1; // should remain at 1 for now, some Refine functions do not support this parameter
  const int stepXY = 1;
};

} // namespace depthMap
} // namespace aliceVision
