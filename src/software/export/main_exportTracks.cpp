// This file is part of the AliceVision project.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/matching/IndMatch.hpp>
#include <aliceVision/matching/io.hpp>
#include <aliceVision/image/all.hpp>
#include <aliceVision/feature/feature.hpp>
#include <aliceVision/track/Track.hpp>
#include <aliceVision/sfm/sfm.hpp>
#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/feature/svgVisualization.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/cmdline.hpp>

#include <software/utils/sfmHelper/sfmIOHelper.hpp>

#include <dependencies/vectorGraphics/svgDrawer.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>

using namespace aliceVision;
using namespace aliceVision::feature;
using namespace aliceVision::matching;
using namespace aliceVision::sfm;
using namespace aliceVision::track;
using namespace svg;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char ** argv)
{
  // command-line parameters

  std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
  std::string sfmDataFilename;
  std::string outputFolder;
  std::string featuresFolder;
  std::string matchesFolder;
  std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);

  po::options_description allParams("AliceVision exportTracks");

  po::options_description requiredParams("Required parameters");
  requiredParams.add_options()
    ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
      "SfMData file.")
    ("output,o", po::value<std::string>(&outputFolder)->required(),
      "Output path for tracks.")
    ("featuresFolder,f", po::value<std::string>(&featuresFolder)->required(),
      "Path to a folder containing the extracted features.")
    ("matchesFolder,m", po::value<std::string>(&matchesFolder)->required(),
      "Path to a folder in which computed matches are stored.");

  po::options_description optionalParams("Optional parameters");
  optionalParams.add_options()
    ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName),
      feature::EImageDescriberType_informations().c_str());

  po::options_description logParams("Log parameters");
  logParams.add_options()
    ("verboseLevel,v", po::value<std::string>(&verboseLevel)->default_value(verboseLevel),
      "verbosity level (fatal,  error, warning, info, debug, trace).");

  allParams.add(requiredParams).add(optionalParams).add(logParams);

  po::variables_map vm;
  try
  {
    po::store(po::parse_command_line(argc, argv, allParams), vm);

    if(vm.count("help") || (argc == 1))
    {
      ALICEVISION_COUT(allParams);
      return EXIT_SUCCESS;
    }
    po::notify(vm);
  }
  catch(boost::program_options::required_option& e)
  {
    ALICEVISION_CERR("ERROR: " << e.what());
    ALICEVISION_COUT("Usage:\n\n" << allParams);
    return EXIT_FAILURE;
  }
  catch(boost::program_options::error& e)
  {
    ALICEVISION_CERR("ERROR: " << e.what());
    ALICEVISION_COUT("Usage:\n\n" << allParams);
    return EXIT_FAILURE;
  }

  ALICEVISION_COUT("Program called with the following parameters:");
  ALICEVISION_COUT(vm);

  // set verbose level
  system::Logger::get()->setLogLevel(verboseLevel);

  if(outputFolder.empty())
  {
    ALICEVISION_LOG_ERROR("It is an invalid output folder");
    return EXIT_FAILURE;
  }

  // read SfM Scene (image view names)
  SfMData sfmData;
  if(!Load(sfmData, sfmDataFilename, ESfMData(VIEWS|INTRINSICS)))
  {
    ALICEVISION_LOG_ERROR("The input SfMData file '" << sfmDataFilename << "' cannot be read.");
    return EXIT_FAILURE;
  }
  
  // get imageDescriberMethodTypes
  std::vector<EImageDescriberType> describerMethodTypes = EImageDescriberType_stringToEnums(describerTypesName);

  // read the features
  feature::FeaturesPerView featuresPerView;
  if(!sfm::loadFeaturesPerView(featuresPerView, sfmData, featuresFolder, describerMethodTypes))
  {
    ALICEVISION_LOG_ERROR("Invalid features");
    return EXIT_FAILURE;
  }

  // read the matches
  matching::PairwiseMatches pairwiseMatches;
  if(!sfm::loadPairwiseMatches(pairwiseMatches, sfmData, matchesFolder, describerMethodTypes))
  {
    ALICEVISION_LOG_ERROR("Invalid matches file");
    return EXIT_FAILURE;
  }

  const std::size_t viewCount = sfmData.GetViews().size();
  ALICEVISION_LOG_INFO("# views: " << viewCount);

  // compute tracks from matches
  track::TracksMap mapTracks;
  {
    const aliceVision::matching::PairwiseMatches& map_Matches = pairwiseMatches;
    track::TracksBuilder tracksBuilder;
    tracksBuilder.Build(map_Matches);
    tracksBuilder.Filter();
    tracksBuilder.ExportToSTL(mapTracks);

    ALICEVISION_LOG_INFO("# tracks: " << tracksBuilder.NbTracks());
  }

  // for each pair, export the matches
  fs::create_directory(outputFolder);
  boost::progress_display myProgressBar( (viewCount*(viewCount-1)) / 2.0 , std::cout, "Export pairwise tracks\n");

  for(std::size_t I = 0; I < viewCount; ++I)
  {
    auto itI = sfmData.GetViews().begin();
    std::advance(itI, I);

    const View* viewI = itI->second.get();

    for(std::size_t J = I+1; J < viewCount; ++J, ++myProgressBar)
    {
      auto itJ = sfmData.GetViews().begin();
      std::advance(itJ, J);

      const View* viewJ = itJ->second.get();

      const std::string& viewImagePathI = viewI->getImagePath();
      const std::string& viewImagePathJ = viewJ->getImagePath();

      const std::pair<std::size_t, std::size_t> dimImageI = std::make_pair(viewI->getWidth(), viewI->getHeight());
      const std::pair<std::size_t, std::size_t> dimImageJ = std::make_pair(viewJ->getWidth(), viewJ->getHeight());

      // get common tracks between view I and J
      track::TracksMap mapTracksCommon;
      std::set<std::size_t> setImageIndex;

      setImageIndex.insert(viewI->getViewId());
      setImageIndex.insert(viewJ->getViewId());

      TracksUtilsMap::GetCommonTracksInImages(setImageIndex, mapTracks, mapTracksCommon);

      if(mapTracksCommon.empty())
      {
        ALICEVISION_LOG_TRACE("no common tracks for pair (" << viewI->getViewId() << ", " << viewJ->getViewId() << ")");
      }
      else
      {
        svgDrawer svgStream(dimImageI.first + dimImageJ.first, max(dimImageI.second, dimImageJ.second));
        svgStream.drawImage(viewImagePathI, dimImageI.first, dimImageI.second);
        svgStream.drawImage(viewImagePathJ, dimImageJ.first, dimImageJ.second, dimImageI.first);

        // draw link between features :
        for (track::TracksMap::const_iterator tracksIt = mapTracksCommon.begin();
          tracksIt != mapTracksCommon.end(); ++tracksIt)
        {
          const feature::EImageDescriberType descType = tracksIt->second.descType;
          assert(descType != feature::EImageDescriberType::UNINITIALIZED);
          track::Track::FeatureIdPerView::const_iterator obsIt = tracksIt->second.featPerView.begin();

          const PointFeatures& featuresI = featuresPerView.getFeatures(viewI->getViewId(), descType);
          const PointFeatures& featuresJ = featuresPerView.getFeatures(viewJ->getViewId(), descType);

          const PointFeature& imaA = featuresI[obsIt->second];
          ++obsIt;
          const PointFeature& imaB = featuresJ[obsIt->second];

          svgStream.drawLine(imaA.x(), imaA.y(), imaB.x()+dimImageI.first, imaB.y(), svgStyle().stroke("green", 2.0));
        }

        // draw features (in two loop, in order to have the features upper the link, svg layer order):
        for(track::TracksMap::const_iterator tracksIt = mapTracksCommon.begin();
          tracksIt != mapTracksCommon.end(); ++ tracksIt)
        {
          const feature::EImageDescriberType descType = tracksIt->second.descType;
          assert(descType != feature::EImageDescriberType::UNINITIALIZED);
          track::Track::FeatureIdPerView::const_iterator obsIt = tracksIt->second.featPerView.begin();

          const PointFeatures& featuresI = featuresPerView.getFeatures(viewI->getViewId(), descType);
          const PointFeatures& featuresJ = featuresPerView.getFeatures(viewJ->getViewId(), descType);

          const PointFeature& imaA = featuresI[obsIt->second];
          ++obsIt;
          const PointFeature& imaB = featuresJ[obsIt->second];

          const std::string featColor = describerTypeColor(descType);

          svgStream.drawCircle(imaA.x(), imaA.y(), 3.0, svgStyle().stroke(featColor, 2.0));
          svgStream.drawCircle(imaB.x() + dimImageI.first,imaB.y(), 3.0, svgStyle().stroke(featColor, 2.0));
        }

        fs::path outputFilename = fs::path(outputFolder) / std::string(std::to_string(viewI->getViewId()) + "_" + std::to_string(viewJ->getViewId()) + "_" + std::to_string(mapTracksCommon.size()) + ".svg");

        ofstream svgFile(outputFilename.string());
        svgFile << svgStream.closeSvgFile().str();
      }
    }
  }
  return EXIT_SUCCESS;
}
