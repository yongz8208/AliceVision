// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfm/pipeline/panorama/ReconstructionEngine_sequentialPanorama.hpp>

#include <aliceVision/track/tracksUtils.hpp>

#include <aliceVision/sfmData/CameraPose.hpp>
#include <aliceVision/system/Logger.hpp>

#include <random>

namespace aliceVision {
namespace sfm {

aliceVision::EstimationStatus robustRotationEstimationAC(const Mat &x1, const Mat &x2, std::mt19937 &randomNumberGenerator,  Mat3 &R, std::vector<std::size_t> &vec_inliers);

ReconstructionEngine_sequentialPanorama::ReconstructionEngine_sequentialPanorama(
  const sfmData::SfMData& sfmData,
  const Params& params,
  const std::string& outputFolder,
  const std::string& loggingFile)
  : ReconstructionEngine(sfmData, outputFolder),
    _params(params)
{
}

void ReconstructionEngine_sequentialPanorama::initializePyramidScoring()
{
  // update cache values
  if (_pyramidWeights.size() != _params.pyramidDepth)
  {
    _pyramidWeights.resize(_params.pyramidDepth);
    std::size_t maxWeight = 0;
    for(std::size_t level = 0; level < _params.pyramidDepth; ++level)
    {
      std::size_t nbCells = Square(std::pow(_params.pyramidBase, level+1));
      // We use a different weighting strategy than [Schonberger 2016].
      // They use w = 2^l with l={1...L} (even if there is a typo in the text where they say to use w=2^{2*l}.
      // We prefer to give more importance to the first levels of the pyramid, so:
      // w = 2^{L-l} with L the number of levels in the pyramid.
      _pyramidWeights[level] = std::pow(2.0, (_params.pyramidDepth-(level+1)));
      maxWeight += nbCells * _pyramidWeights[level];
    }
    _pyramidThreshold = maxWeight * 0.2;
  }
}

std::size_t ReconstructionEngine_sequentialPanorama::fuseMatchesIntoTracks()
{
  // compute tracks from matches
  track::TracksBuilder tracksBuilder;

  // list of features matches for each couple of images
  const aliceVision::matching::PairwiseMatches& matches = *_pairwiseMatches;


  tracksBuilder.build(matches);
  tracksBuilder.filter(_params.filterTrackForks, _params.minInputTrackLength);
  tracksBuilder.exportToSTL(_map_tracks);

  // Init tracksPerView to have an entry in the map for each view (even if there is no track at all)
  for(const auto& viewIt: _sfmData.views)
  {
      // create an entry in the map
      _map_tracksPerView[viewIt.first];
  }


  track::computeTracksPerView(_map_tracks, _map_tracksPerView);
  computeTracksPyramidPerView(_map_tracksPerView, _map_tracks, _sfmData.views, *_featuresPerView, _params.pyramidBase, _params.pyramidDepth, _map_featsPyramidPerView);

  return _map_tracks.size();
}

std::size_t ReconstructionEngine_sequentialPanorama::computeCandidateImageScore(IndexT viewId, const std::vector<std::size_t>& trackIds) const
{
  std::size_t score = 0;

  // The number of cells of the pyramid grid represent the score
  // and ensure a proper repartition of features in images.
  const auto& featsPyramid = _map_featsPyramidPerView.at(viewId);
  for(std::size_t level = 0; level < _params.pyramidDepth; ++level)
  {
    std::set<std::size_t> featIndexes; // Set of grid cell indexes in the pyramid
    for(IndexT trackId: trackIds)
    {
      std::size_t pyramidIndex = featsPyramid.at(trackId * _params.pyramidDepth + level);
      featIndexes.insert(pyramidIndex);
    }
    score += featIndexes.size() * _pyramidWeights[level];
  }

  return score;
}


bool ReconstructionEngine_sequentialPanorama::findNextPair(std::pair<IndexT, IndexT> & pair, Mat3 & foundRotation, const std::set<IndexT> & reconstructedViews, const std::set<IndexT> & availableViews)
{  
  Mat3 bestR;
  size_t bestScore = 0;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < _pairwiseMatches->size(); ++i)
  {
    matching::PairwiseMatches::const_iterator iter = _pairwiseMatches->begin();
    std::advance(iter, i);

    const Pair current_pair = iter->first;
    const IndexT I = std::min(current_pair.first, current_pair.second);
    const IndexT J = std::max(current_pair.first, current_pair.second);

    if (reconstructedViews.size() == 0)
    {
      if (!(availableViews.count(I) && availableViews.count(J)))
      {
        continue;
      }
    }
    else 
    {
      ///If at least a reconstructed view is available, we want one view to be a reconstructed view, and the other a new view
      if (!(availableViews.count(I) && reconstructedViews.count(J)))
      {
        if (!(availableViews.count(J) && reconstructedViews.count(I)))
        {
          continue;
        }
      }
    }

    const std::shared_ptr<sfmData::View> viewI = _sfmData.getViews().at(I);
    const sfmData::Intrinsics::const_iterator iterIntrinsic_I = _sfmData.getIntrinsics().find(viewI->getIntrinsicId());
    const std::shared_ptr<sfmData::View> viewJ = _sfmData.getViews().at(J);
    const sfmData::Intrinsics::const_iterator iterIntrinsic_J = _sfmData.getIntrinsics().find(viewJ->getIntrinsicId());

    const std::shared_ptr<camera::Pinhole> camI = std::dynamic_pointer_cast<camera::Pinhole>(iterIntrinsic_I->second);
    const std::shared_ptr<camera::Pinhole> camJ = std::dynamic_pointer_cast<camera::Pinhole>(iterIntrinsic_J->second);
    if (camI == nullptr || camJ == nullptr)
    {
      continue;
    }

    aliceVision::track::TracksMap map_tracksCommon;
    const std::set<size_t> set_imageIndex= {I, J};
    track::getCommonTracksInImagesFast(set_imageIndex, _map_tracks, _map_tracksPerView, map_tracksCommon);

    const size_t n = map_tracksCommon.size();
    Mat xI(3,n), xJ(3,n);
    
    size_t cptIndex = 0;
    for (aliceVision::track::TracksMap::const_iterator iterT = map_tracksCommon.begin(); iterT != map_tracksCommon.end(); ++iterT, ++cptIndex)
    {
      auto iter = iterT->second.featPerView.begin();
      
      const size_t i = iter->second;
      const size_t j = (++iter)->second;
      
      const auto& viewI = _featuresPerView->getFeatures(I, iterT->second.descType); 
      const auto& viewJ = _featuresPerView->getFeatures(J, iterT->second.descType);
      
      Vec2 featI = viewI[i].coords().cast<double>();
      Vec2 featJ = viewJ[j].coords().cast<double>();

      const Vec3 bearingVectorI = camI->toUnitSphere(camI->removeDistortion(camI->ima2cam(featI)));
      const Vec3 bearingVectorJ = camJ->toUnitSphere(camJ->removeDistortion(camJ->ima2cam(featJ)));

      xI.col(cptIndex) = bearingVectorI;
      xJ.col(cptIndex) = bearingVectorJ;
    }

    std::vector<std::size_t> vec_inliers;
    Mat3 relativeRotation;

    std::mt19937 lrandom = _randomNumberGenerator;
    const auto status = robustRotationEstimationAC(xI, xJ, lrandom, relativeRotation, vec_inliers);
    if (!status.isValid && !status.hasStrongSupport) {
      continue;
    }

#pragma omp critical
    if (vec_inliers.size() > bestScore)
    {
      bestScore = vec_inliers.size();
      foundRotation = relativeRotation;
      pair.first = I;
      pair.second = J;
    }
  }

  if (bestScore == 0)
  {
    return false;
  }

  return true;
}

bool ReconstructionEngine_sequentialPanorama::incrementalReconstruction()
{
  std::set<IndexT> validTracks;

  for(const auto& viewIt: _sfmData.views)
  {
    IndexT viewId = viewIt.first;

    if (_sfmData.isPoseAndIntrinsicDefined(viewId))
    {
      const aliceVision::track::TrackIdSet& set_tracksIds = _map_tracksPerView.at(viewId);
      validTracks.insert(set_tracksIds.begin(), set_tracksIds.end());
    }
  }

  return true;
}

bool ReconstructionEngine_sequentialPanorama::process()
{
  initializePyramidScoring();

  if(fuseMatchesIntoTracks() == 0)
  {
    ALICEVISION_LOG_ERROR("No valid tracks.");
    return false;
  }

  std::set<IndexT> availableViews;
  std::set<IndexT> reconstructedViews;

  // List Views that support valid intrinsic (view that could be used for Essential matrix computation)
  for(const auto& it : _sfmData.getViews())
  {
    std::shared_ptr<sfmData::View> v = it.second;
    if (_sfmData.getIntrinsics().count(v->getIntrinsicId()) && _sfmData.getIntrinsics().at(v->getIntrinsicId())->isValid())
    {
      availableViews.insert(v->getViewId());
    }
  }

  Mat3 R;
  std::pair<IndexT, IndexT> firstPair;
  bool ret = findNextPair(firstPair, R, reconstructedViews, availableViews);
  if (!ret)
  {
    return false;
  }

  reconstructedViews.insert(firstPair.first);
  reconstructedViews.insert(firstPair.second);
  availableViews.erase(firstPair.first);
  availableViews.erase(firstPair.second);

  //Build initial rotation
  const geometry::Pose3& initPose1 = geometry::Pose3(Mat3::Identity(), Vec3::Zero());
  const geometry::Pose3& initPose2 = geometry::Pose3(R, Vec3::Zero());
  _sfmData.setPose(_sfmData.getView(firstPair.first), sfmData::CameraPose(initPose1));
  _sfmData.setPose(_sfmData.getView(firstPair.second), sfmData::CameraPose(initPose2));

  while (1)
  {
    Mat3 R;
    std::pair<IndexT, IndexT> nextPair;
    bool ret = findNextPair(nextPair, R, reconstructedViews, availableViews);
    if (!ret)
    {
      return false;
    }

    reconstructedViews.insert(nextPair.first);
    reconstructedViews.insert(nextPair.second);
    availableViews.erase(nextPair.first);
    availableViews.erase(nextPair.second);

    std::cout << nextPair.first << " " << nextPair.second << std::endl;
  }

  return true;
}

} // namespace sfm
} // namespace aliceVision
