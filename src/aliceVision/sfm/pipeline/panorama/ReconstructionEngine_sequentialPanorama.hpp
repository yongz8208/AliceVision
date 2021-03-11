// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/sfm/pipeline/ReconstructionEngine.hpp>
#include <aliceVision/sfm/pipeline/pairwiseMatchesIO.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/feature/FeaturesPerView.hpp>
#include <aliceVision/track/TracksBuilder.hpp>

namespace aliceVision {
namespace sfm {


/**
 * @brief Sequential SfM Pipeline Reconstruction Engine.
 */
class ReconstructionEngine_sequentialPanorama : public ReconstructionEngine
{
public:
  struct Params
  {
    const int pyramidBase = 2;
    const int pyramidDepth = 5;
    bool filterTrackForks = true;
    int minInputTrackLength = 2;
  };

public:

  ReconstructionEngine_sequentialPanorama(const sfmData::SfMData& sfmData,
                                     const Params& params,
                                     const std::string& outputFolder,
                                     const std::string& loggingFile = "");

  void setFeatures(feature::FeaturesPerView* featuresPerView)
  {
    _featuresPerView = featuresPerView;
  }

  void setMatches(matching::PairwiseMatches* pairwiseMatches)
  {
    _pairwiseMatches = pairwiseMatches;
  }

  /**
   * @brief Process the entire incremental reconstruction
   * @return true if done
   */
  virtual bool process();

private:
  std::size_t fuseMatchesIntoTracks();
  void initializePyramidScoring();
  bool findNextPair(std::pair<IndexT, IndexT> & pair, Mat3 & foundRotation, const std::set<IndexT> & reconstructedViews, const std::set<IndexT> & availableViews);
  bool incrementalReconstruction();
  bool bundleAdjustment(std::set<IndexT>& reconstructedViews);
  
  /**
   * @brief Compute a score of the view for a subset of features. This is
   *        used for the next best view choice.
   *
   * The score is based on a pyramid which allows to compute a weighting
   * strategy to promote a good repartition in the image (instead of relying
   * only on the number of features).
   * Inspired by [Schonberger 2016]:
   * "Structure-from-Motion Revisited", Johannes L. Schonberger, Jan-Michael Frahm
   * 
   * http://people.inf.ethz.ch/jschoenb/papers/schoenberger2016sfm.pdf
   * We don't use the same weighting strategy. The weighting choice
   * is not justified in the paper.
   *
   * @param[in] viewId: the ID of the view
   * @param[in] trackIds: set of track IDs contained in viewId
   * @return the computed score
   */
  std::size_t computeCandidateImageScore(IndexT viewId, const std::vector<std::size_t>& trackIds) const;

private:
  // Parameters
  Params _params;

  // Data providers
  feature::FeaturesPerView* _featuresPerView;
  matching::PairwiseMatches* _pairwiseMatches;

  track::TracksMap _map_tracks;
  track::TracksPerView _map_tracksPerView;
  track::TracksPyramidPerView _map_featsPyramidPerView;

  /// internal cache of precomputed values for the weighting of the pyramid levels
  std::vector<int> _pyramidWeights;
  int _pyramidThreshold;
};

} // namespace sfm
} // namespace aliceVision

