// This file is part of the AliceVision project.
// Copyright (c) 2019 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ReconstructionEngine.hpp"

#include <aliceVision/feature/RegionsPerView.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>


namespace aliceVision {
namespace sfm {

void computeTracksPyramidPerView(
    const track::TracksPerView& tracksPerView,
    const track::TracksMap& map_tracks,
    const sfmData::Views& views,
    const feature::FeaturesPerView& featuresProvider,
    const std::size_t pyramidBase,
    const std::size_t pyramidDepth,
    track::TracksPyramidPerView& tracksPyramidPerView)
{

  std::vector<std::size_t> widthPerLevel(pyramidDepth);
  std::vector<std::size_t> startPerLevel(pyramidDepth);
  std::size_t start = 0;

  for(std::size_t level = 0; level < pyramidDepth; ++level)
  {
    startPerLevel[level] = start;
    widthPerLevel[level] = std::pow(pyramidBase, level+1);
    start += Square(widthPerLevel[level]);
  }

  tracksPyramidPerView.reserve(tracksPerView.size());
  for(const auto& viewTracks: tracksPerView)
  {
    auto& trackPyramid = tracksPyramidPerView[viewTracks.first];
    trackPyramid.reserve(viewTracks.second.size() * pyramidDepth);
  }

  for(const auto& viewTracks: tracksPerView)
  {
    const auto viewId = viewTracks.first;
    auto & tracksPyramidIndex = tracksPyramidPerView[viewId];
    const sfmData::View & view = *views.at(viewId).get();

    //Compute grid size
    std::vector<double> cellWidthPerLevel(pyramidDepth);
    std::vector<double> cellHeightPerLevel(pyramidDepth);
    for(std::size_t level = 0; level < pyramidDepth; ++level)
    {
      cellWidthPerLevel[level] = (double)view.getWidth() / (double)widthPerLevel[level];
      cellHeightPerLevel[level] = (double)view.getHeight() / (double)widthPerLevel[level];
    }

    for(std::size_t i = 0; i < viewTracks.second.size(); ++i)
    {
      const std::size_t trackId = viewTracks.second[i];
      const track::Track& track = map_tracks.at(trackId);
      const std::size_t featIndex = track.featPerView.at(viewId);
      const auto& feature = featuresProvider.getFeatures(viewId, track.descType)[featIndex]; 
      
      //Store each feature inside one cell for each pyramid level
      for(std::size_t level = 0; level < pyramidDepth; ++level)
      {
        std::size_t xCell = std::floor(std::max(feature.x(), 0.0f) / cellWidthPerLevel[level]);
        std::size_t yCell = std::floor(std::max(feature.y(), 0.0f) / cellHeightPerLevel[level]);
        xCell = std::min(xCell, widthPerLevel[level] - 1);
        yCell = std::min(yCell, widthPerLevel[level] - 1);
        const std::size_t levelIndex = xCell + yCell * widthPerLevel[level];

        assert(levelIndex < Square(widthPerLevel[level]));
        tracksPyramidIndex[trackId * pyramidDepth + level] = startPerLevel[level] + levelIndex;
      }
    }
  }
}


void retrieveMarkersId(sfmData::SfMData& sfmData)
{
    std::set<feature::EImageDescriberType> allMarkerDescTypes;
#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_CCTAG)
    allMarkerDescTypes.insert(feature::EImageDescriberType::CCTAG3);
    allMarkerDescTypes.insert(feature::EImageDescriberType::CCTAG4);
#endif
    if (allMarkerDescTypes.empty())
        return;

    std::set<feature::EImageDescriberType> usedDescTypes = sfmData.getLandmarkDescTypes();

    std::vector<feature::EImageDescriberType> markerDescTypes;
    std::set_intersection(allMarkerDescTypes.begin(), allMarkerDescTypes.end(),
        usedDescTypes.begin(), usedDescTypes.end(),
        std::back_inserter(markerDescTypes));

    std::set<feature::EImageDescriberType> markerDescTypes_set(markerDescTypes.begin(), markerDescTypes.end());

    if(markerDescTypes.empty())
        return;

    // load the corresponding view regions
    feature::RegionsPerView regionPerView;
    std::set<IndexT> filter;
    // It could be optimized by loading only the minimal number of desc files,
    // but as we are only retrieving them for markers, the performance impact is limited.
    if (!sfm::loadRegionsPerView(regionPerView, sfmData, sfmData.getFeaturesFolders(), markerDescTypes, filter))
    {
        ALICEVISION_THROW_ERROR("Error while loading markers regions.");
    }
    for (auto& landmarkIt : sfmData.getLandmarks())
    {
        auto& landmark = landmarkIt.second;
        if (landmark.observations.empty())
            continue;
        if (markerDescTypes_set.find(landmark.descType) == markerDescTypes_set.end())
            continue;
        landmark.rgb = image::BLACK;

        const auto obs = landmark.observations.begin();
        const feature::Regions& regions = regionPerView.getRegions(obs->first, landmark.descType);
        const feature::CCTAG_Regions& cctagRegions = dynamic_cast<const feature::CCTAG_Regions&>(regions);
        const auto& d = cctagRegions.Descriptors()[obs->second.id_feat];
        for (int i = 0; i < d.size(); ++i)
        {
            if (d[i] == 255)
            {
                ALICEVISION_LOG_TRACE("Found marker: " << i << " (landmarkId: " << landmarkIt.first << ").");
                landmark.rgb.r() = i;
                break;
            }
        }
    }
}


} // namespace sfm
} // namespace aliceVision
