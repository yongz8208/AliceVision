// This file is part of the AliceVision project.
// Copyright (c) 2015 AliceVision contributors.
// Copyright (c) 2012 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>
#include <aliceVision/sfm/sfm.hpp>

#include <aliceVision/sfm/pipeline/regionsIO.hpp>
#include <aliceVision/feature/imageDescriberCommon.hpp>
#include <aliceVision/system/Timer.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/types.hpp>
#include <aliceVision/config.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 1

using namespace aliceVision;

namespace po = boost::program_options;
namespace fs = boost::filesystem;


int aliceVision_main(int argc, char **argv)
{
  // command-line parameters
  std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
  std::string sfmDataFilename;
  std::string outputSfM;

  std::vector<std::string> featuresFolders;
  std::vector<std::string> matchesFolders;
  
  std::string describerTypesName = feature::EImageDescriberType_enumToString(feature::EImageDescriberType::SIFT);

  int randomSeed = std::mt19937::default_seed;

  po::options_description allParams(
    "Matching filtering\n"
    "AliceVision graphFiltering");

  po::options_description requiredParams("Required parameters");
  requiredParams.add_options()
    ("input,i", po::value<std::string>(&sfmDataFilename)->required(),
      "SfMData file.")
    ("output,o", po::value<std::string>(&outputSfM)->required(),
      "Path to the output SfMData file.")
    ;

  po::options_description optionalParams("Optional parameters");
  optionalParams.add_options()
    ("featuresFolders,f", po::value<std::vector<std::string>>(&featuresFolders)->multitoken(),
      "Path to folder(s) containing the extracted features.")
    ("matchesFolders,m", po::value<std::vector<std::string>>(&matchesFolders)->multitoken(),
      "Path to folder(s) in which computed matches are stored.")
    ("describerTypes,d", po::value<std::string>(&describerTypesName)->default_value(describerTypesName),
      feature::EImageDescriberType_informations().c_str())
    ("randomSeed", po::value<int>(&randomSeed)->default_value(randomSeed),
      "This seed value will generate a sequence using a linear random generator. Set -1 to use a random seed.")
    ;

  po::options_description logParams("Log parameters");
  logParams.add_options()
    ("verboseLevel,v", po::value<std::string>(&verboseLevel)->default_value(verboseLevel),
      "verbosity level (fatal, error, warning, info, debug, trace).");

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

  // load input SfMData scene
  sfmData::SfMData sfmData;
  if(!sfmDataIO::Load(sfmData, sfmDataFilename, sfmDataIO::ESfMData::ALL))
  {
    ALICEVISION_LOG_ERROR("The input SfMData file '" + sfmDataFilename + "' cannot be read.");
    return EXIT_FAILURE;
  }

  // get imageDescriber type
  const std::vector<feature::EImageDescriberType> describerTypes = feature::EImageDescriberType_stringToEnums(describerTypesName);

  // features reading
  feature::FeaturesPerView featuresPerView;
  if(!sfm::loadFeaturesPerView(featuresPerView, sfmData, featuresFolders, describerTypes))
  {
    ALICEVISION_LOG_ERROR("Invalid features.");
    return EXIT_FAILURE;
  }
  
  // matches reading
  matching::PairwiseMatches pairwiseMatches;
  if(!sfm::loadPairwiseMatches(pairwiseMatches, sfmData, matchesFolders, describerTypes, 0, 0, false))
  {
    ALICEVISION_LOG_ERROR("Unable to load matches.");
    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}
