// This file is part of the AliceVision project.
// Copyright (c) 2016 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfm/pipeline/panorama/ReconstructionEngine_sequentialPanorama.hpp>

#include <aliceVision/track/tracksUtils.hpp>
#include <aliceVision/sfm/BundleAdjustmentSymbolicCeres.hpp>

#include <aliceVision/sfmData/CameraPose.hpp>
#include <aliceVision/system/Logger.hpp>

#include <chrono>
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


bool ReconstructionEngine_sequentialPanorama::findFirstPair(std::pair<IndexT, IndexT> & pair, Mat3 & foundRotation, const std::set<IndexT> & availableViews)
{  
  Mat3 bestR;
  size_t bestScore = 0;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < _pairwiseMatches->size(); ++i)
  {
    matching::PairwiseMatches::const_iterator iter = _pairwiseMatches->begin();
    std::advance(iter, i);

    const Pair current_pair = iter->first;
    IndexT I = std::min(current_pair.first, current_pair.second);
    IndexT J = std::max(current_pair.first, current_pair.second);

    if (!(availableViews.count(I) && availableViews.count(J)))
    {
      continue;
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

  if (bestScore < 30)
  {
    return false;
  }

  return true;
}

IndexT ReconstructionEngine_sequentialPanorama::findNextBestImage(const std::set<IndexT> & availableViews)
{
  size_t bestScore = 0;
  IndexT bestId = UndefinedIndexT;

  for (IndexT idview : availableViews)
  {
    const aliceVision::track::TrackIdSet& set_tracksIds = _map_tracksPerView.at(idview);
    if (set_tracksIds.empty())
    {
      continue;
    }

    std::vector<std::size_t> validTrackIds;
    std::set_intersection(set_tracksIds.begin(), set_tracksIds.end(), _reconstructed_trackId.begin(), _reconstructed_trackId.end(), std::back_inserter(validTrackIds));

    if (validTrackIds.size() > bestScore)
    {
      bestScore = validTrackIds.size();
      bestId = idview;
    }
  }

  return bestId;
}

bool ReconstructionEngine_sequentialPanorama::buildLandmarks(sfmData::View & view)
{
  const sfmData::Intrinsics::const_iterator iterIntrinsic_I = _sfmData.getIntrinsics().find(view.getIntrinsicId());
  const std::shared_ptr<camera::Pinhole> cam = std::dynamic_pointer_cast<camera::Pinhole>(iterIntrinsic_I->second);
  if (cam == nullptr)
  {
    return false;
  }

  const track::TrackIdSet & tracks = _map_tracksPerView[view.getViewId()];

  const sfmData::CameraPose & pose = _sfmData.getPose(view);
  
  for (IndexT trackId : tracks)
  {
    const track::Track & t = _map_tracks[trackId];
    const IndexT & featureId = t.featPerView.at(view.getViewId());

    const feature::PointFeatures & features = _featuresPerView->getFeatures(view.getViewId(), t.descType); 
    
    Vec2 coords = features[featureId].coords().cast<double>();
    Vec3 pt = cam->toUnitSphere(cam->removeDistortion(cam->ima2cam(coords)));

    if (_reconstructed_trackId.find(trackId) == _reconstructed_trackId.end())
    {
      sfmData::Landmark l(t.descType);
            
      double norm = pt.z() + pt.norm();

      l.X.x() = pt.x() / norm;
      l.X.y() = pt.y() / norm;
      l.X.z() = 0.0;

      l.referenceView = view.getViewId();

      sfmData::Observation obs(coords, featureId, 1.0);
      l.observations[view.getViewId()] = obs;

      _sfmData.getLandmarks()[trackId] = l;
      _reconstructed_trackId.insert(trackId);
    }
    else 
    {
      sfmData::Landmark & l = _sfmData.getLandmarks()[trackId];
      if (l.observations.size() == 0) continue;

      sfmData::Observation obs(coords, featureId, 1.0);
      l.observations[view.getViewId()] = obs;
    }
  }

  return true;
}

bool ReconstructionEngine_sequentialPanorama::estimateView(sfmData::View & v)
{
  const sfmData::Intrinsics::const_iterator iterIntrinsicCurrent = _sfmData.getIntrinsics().find(v.getIntrinsicId());
  const std::shared_ptr<camera::Pinhole> camCurrent = std::dynamic_pointer_cast<camera::Pinhole>(iterIntrinsicCurrent->second);
  if (camCurrent == nullptr)
  {
    return false;
  }

  const aliceVision::track::TrackIdSet& set_tracksIds = _map_tracksPerView.at(v.getViewId());
  if (set_tracksIds.empty())
  {
    return false;
  }

  std::vector<std::size_t> validTrackIds;
  std::set_intersection(set_tracksIds.begin(), set_tracksIds.end(), _reconstructed_trackId.begin(), _reconstructed_trackId.end(), std::back_inserter(validTrackIds));

  
  const size_t n = validTrackIds.size();
  Mat xI(3,n), xJ(3,n);

  int pos = 0;
  for (IndexT trackId : validTrackIds)
  {
    const track::Track & t = _map_tracks[trackId];

    const sfmData::Landmark & l = _sfmData.getLandmarks()[trackId];

    IndexT refViewId = l.referenceView;
    const sfmData::View & refView = _sfmData.getView(refViewId);

    const IndexT & referenceFeatureId = t.featPerView.at(refViewId);
    const IndexT & currentFeatureId = t.featPerView.at(v.getViewId());

    const sfmData::Intrinsics::const_iterator iterIntrinsicReference = _sfmData.getIntrinsics().find(refView.getIntrinsicId());
    const std::shared_ptr<camera::Pinhole> camReference = std::dynamic_pointer_cast<camera::Pinhole>(iterIntrinsicReference->second);

    const geometry::Pose3 refPose = _sfmData.getPose(refView).getTransform();

    const feature::PointFeatures & featuresReferenceView = _featuresPerView->getFeatures(refViewId, t.descType); 
    const feature::PointFeatures & featuresCurrentView = _featuresPerView->getFeatures(v.getViewId(), t.descType); 
    
    Vec2 referenceCoords = featuresReferenceView[referenceFeatureId].coords().cast<double>();
    Vec2 currentCoords = featuresCurrentView[currentFeatureId].coords().cast<double>();


    double u = l.X.x();
    double v = l.X.y();

    double u2 = u * u;
    double v2 = v * v;
    double r2 = u2 + v2;

    double lambda = 2.0 / (1.0 + r2);
    Vec3 X;
    X.x() = u * lambda;
    X.y() = v * lambda;
    X.z() = lambda - 1.0;

    Vec3 rpt = refPose.rotation().transpose() * X;
    Vec3 cpt = camCurrent->toUnitSphere(camCurrent->removeDistortion(camCurrent->ima2cam(currentCoords)));    

    xI.col(pos) = rpt;
    xJ.col(pos) = cpt;
    pos++;
  }


  std::vector<std::size_t> vec_inliers;
  Mat3 relativeRotation;

  std::mt19937 lrandom = _randomNumberGenerator;
  const auto status = robustRotationEstimationAC(xI, xJ, lrandom, relativeRotation, vec_inliers);
  if (!status.isValid && !status.hasStrongSupport) {
    return false;
  }

  if (vec_inliers.size() < 30)
  {
    return false;
  }

  const geometry::Pose3& initPose = geometry::Pose3(relativeRotation, Vec3::Zero());
  _sfmData.setPose(v, sfmData::CameraPose(initPose));

  std::cout << vec_inliers.size() << std::endl;


  return true;
}

void ReconstructionEngine_sequentialPanorama::removeOutliers()
{
  size_t pos = 0;

  for (auto & l : _sfmData.getLandmarks())
  {
    IndexT refViewId = l.second.referenceView;
    sfmData::View & refView = _sfmData.getView(refViewId);
    Mat3 rRw = _sfmData.getPose(refView).getTransform().rotation();
    

    double u = l.second.X.x();
    double v = l.second.X.y();

    double u2 = u * u;
    double v2 = v * v;
    double r2 = u2 + v2;

    double lambda = 2.0 / (1.0 + r2);
    Vec3 rpt;
    rpt.x() = u * lambda;
    rpt.y() = v * lambda;
    rpt.z() = lambda - 1.0;
    
    sfmData::Observations obsCopy = l.second.observations;
    l.second.observations.clear();

    Vec3 wpt = rRw.transpose() * rpt;

    for (auto & obsIt : obsCopy)
    {
      IndexT curViewId = obsIt.first;
      sfmData::View & curView = _sfmData.getView(curViewId);
      Mat3 cRw = _sfmData.getPose(curView).getTransform().rotation();

      Vec3 cpt = cRw * wpt;
      Vec2 ppt = cpt.head(2) / cpt(2);
  
      const sfmData::Intrinsics::const_iterator iterIntrinsicCurrent = _sfmData.getIntrinsics().find(curView.getIntrinsicId());
      const std::shared_ptr<camera::Pinhole> camCurrent = std::dynamic_pointer_cast<camera::Pinhole>(iterIntrinsicCurrent->second);

      Vec2 ipt = camCurrent->cam2ima(camCurrent->addDistortion(ppt));
      
      Vec2 diff = ipt - obsIt.second.x;
      if (diff.norm() < 1.0)
      {
        l.second.observations[obsIt.first] = obsIt.second;
      }     
    }
  }
}

bool ReconstructionEngine_sequentialPanorama::bundleAdjustment() 
{
  ALICEVISION_LOG_INFO("Bundle adjustment start.");

  auto chronoStart = std::chrono::steady_clock::now();

  BundleAdjustmentSymbolicCeres::CeresOptions options;
  BundleAdjustmentSymbolicCeres::ERefineOptions refineOptions = BundleAdjustmentSymbolicCeres::REFINE_ROTATION | BundleAdjustmentSymbolicCeres::REFINE_INTRINSICS_FOCAL |BundleAdjustmentSymbolicCeres::REFINE_STRUCTURE;


  // enable Sparse solver and local strategy
  if(_sfmData.getPoses().size() > 100)
  {
    options.setSparseBA();
  }
  else
  {
    options.setDenseBA();
  }

  BundleAdjustmentSymbolicCeres BA(options);

  const bool success = BA.adjust(_sfmData, refineOptions);
  if(!success)
  {
    return false;
  }

  removeOutliers();

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

  // List Views that support valid intrinsic (view that could be used for Essential matrix computation)
  std::set<IndexT> availableViews;
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
  bool ret = findFirstPair(firstPair, R, availableViews);
  if (!ret)
  {
    return false;
  }

  std::set<IndexT> reconstructedViews;
  reconstructedViews.insert(firstPair.first);
  reconstructedViews.insert(firstPair.second);
  availableViews.erase(firstPair.first);
  availableViews.erase(firstPair.second);

  //Build initial rotation
  const geometry::Pose3& initPose1 = geometry::Pose3(Mat3::Identity(), Vec3::Zero());
  const geometry::Pose3& initPose2 = geometry::Pose3(R, Vec3::Zero());
  _sfmData.setPose(_sfmData.getView(firstPair.first), sfmData::CameraPose(initPose1, true));
  _sfmData.setPose(_sfmData.getView(firstPair.second), sfmData::CameraPose(initPose2));

  buildLandmarks(_sfmData.getView(firstPair.first));
  buildLandmarks(_sfmData.getView(firstPair.second));

  int pos = 0;

  while (1)
  {
    IndexT nextImage = findNextBestImage(availableViews);
    if (nextImage == UndefinedIndexT)
    {
      break;
    }

    sfmData::View v = _sfmData.getView(nextImage);
    if (!estimateView(v))
    {
      continue;
    }

    reconstructedViews.insert(nextImage);
    availableViews.erase(nextImage);

    bundleAdjustment();
    buildLandmarks(v);

    pos++;

  }  

  bundleAdjustment();

  sfmData::Poses & ps = _sfmData.getPoses();

  Mat3 rRw = ps.begin()->second.getTransform().rotation();

  for (auto & p : ps)
  {
    Mat3 cRw = p.second.getTransform().rotation();

    Mat3 diff = cRw * rRw.transpose();
    Eigen::AngleAxisd aa;
    aa.fromRotationMatrix(diff);

    std::cout << aa.axis().transpose() << std::endl;
  }

  return true;
}

} // namespace sfm
} // namespace aliceVision
