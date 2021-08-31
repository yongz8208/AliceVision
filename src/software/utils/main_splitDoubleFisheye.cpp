// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// Copyright (c) 2016 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/image/io.hpp>
#include <aliceVision/image/Sampler.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <dependencies/vectorGraphics/svgDrawer.hpp>
#include <aliceVision/panorama/sphericalMapping.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp> 

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include <string>
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace oiio = OIIO;


bool splitDualFisheye(std::array<std::string, 2> & pathOutput, const std::string& imagePath, const std::string& outputFolder)
{

  image::Image<image::RGBfColor> imageSource;
  image::readImage(imagePath, imageSource, image::EImageColorSpace::LINEAR);

  oiio::ImageBuf buffer;
  image::getBufferFromImage(imageSource, buffer);

  // all image need to be horizontal
  if(imageSource.Height() > imageSource.Width())
    throw std::runtime_error(std::string("Cannot split dual fisheye from the vertical image '") + imagePath + "'.");

  const int outSide = std::min(imageSource.Height(), imageSource.Width() / 2);
  const int offset = std::abs((imageSource.Width() / 2) - imageSource.Height());
  const int halfOffset = offset / 2;

  image::Image<image::RGBfColor> imageOut(outSide, outSide, true, image::RGBfColor(1.0f));
  oiio::ImageBuf bufferOut;
  image::getBufferFromImage(imageOut, bufferOut);

  for(std::size_t i = 0; i < 2; ++i)
  {
    const int xbegin = i * outSide;
    const int xend = xbegin + outSide;
    int ybegin = 0;
    int yend = outSide;

    const oiio::ROI subImageROI(xbegin, xend, ybegin, yend);
    oiio::ImageBufAlgo::crop(bufferOut, buffer, subImageROI);
    
    boost::filesystem::path path(imagePath);

    bufferOut.get_pixels(subImageROI, oiio::TypeDesc::FLOAT, imageOut.data());

    pathOutput[i] = outputFolder + std::string("/") + path.stem().string() + std::string("_") + std::to_string(i) + path.extension().string();
    image::writeImage(pathOutput[i], imageOut, image::EImageColorSpace::AUTO, image::readImageMetadata(imagePath));
  }
  
  ALICEVISION_LOG_INFO(imagePath + " successfully split");
  return true;
}


int aliceVision_main(int argc, char** argv)
{
  // command-line parameters
  std::string verboseLevel = system::EVerboseLevel_enumToString(system::Logger::getDefaultVerboseLevel());
  std::string inputSfmDataPath; 
  std::string outputSfmDataPath;
  int nbThreads = 3;

  po::options_description allParams("This program is used to extract multiple images from equirectangular or dualfisheye images or image folder\n"
                                    "AliceVision split360Images");

  po::options_description requiredParams("Required parameters");
  requiredParams.add_options()
    ("input,i", po::value<std::string>(&inputSfmDataPath)->required(),
      "Input sfm data.")
    ("output,o", po::value<std::string>(&outputSfmDataPath)->required(),
      "Output sfm data");

  po::options_description optionalParams("Optional parameters");
  optionalParams.add_options()
    ("nbThreads", po::value<int>(&nbThreads)->default_value(nbThreads),
      "Number of threads.");

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
  
  // Load input scene
  sfmData::SfMData sfmData;
  if(!sfmDataIO::Load(sfmData, inputSfmDataPath, sfmDataIO::ESfMData::ALL))
  {
    ALICEVISION_LOG_ERROR("The input SfMData file '" << inputSfmDataPath << "' cannot be read");
    return EXIT_FAILURE;
  }

  boost::filesystem::path path(outputSfmDataPath);
  std::string outputFolder = path.parent_path().string();

  //First make sure that there is only one intrinsic
  if (sfmData.getIntrinsics().size() != 1)
  {
    ALICEVISION_LOG_ERROR("Multiple intrinsics are not allowed in the images list");
    return EXIT_FAILURE;
  }

  //I don't know what to do if there is a rig
  if (sfmData.getRigs().size() != 0)
  {
    ALICEVISION_LOG_ERROR("Input contains a rig and this is not supported");
    return EXIT_FAILURE;
  }

  //Create a rig with 2 images
  sfmData.getRigs()[0] = sfmData::Rig(2);

  //Double intrinsics
  std::shared_ptr<camera::IntrinsicBase> it1(sfmData.getIntrinsics().begin()->second->clone());
  std::shared_ptr<camera::IntrinsicBase> it2(sfmData.getIntrinsics().begin()->second->clone());
  sfmData.getIntrinsics().clear();
  sfmData.getIntrinsics()[0] = it1;
  sfmData.getIntrinsics()[1] = it2;
  it1->setWidth(it1->w() / 2);
  it2->setWidth(it2->w() / 2);
  

  //Dirty trick to allow iterating with openmp
  std::vector<std::shared_ptr<sfmData::View>> views;
  for (auto v: sfmData.getViews())
  {
    views.push_back(v.second);
  }
  
  bool everyThingOK = true;

  std::vector<IndexT> toErase;

#pragma omp parallel for num_threads(nbThreads)
  for(int itView = 0; itView < views.size(); itView++)
  {
    toErase.push_back(views[itView]->getViewId());
    const std::string& imagePath = views[itView]->getImagePath();

    std::array<std::string, 2> pathOutputs;
    if (!splitDualFisheye(pathOutputs, imagePath, outputFolder))
    {
      everyThingOK = false;
      continue;
    }

    #pragma omp critical
    {
      sfmData::View v = *views[itView];

      std::shared_ptr<sfmData::View> vl = std::make_shared<sfmData::View>(v);
      vl->setViewId(itView * 2);
      vl->setRigAndSubPoseId(0, 0);
      vl->setImagePath(pathOutputs[0]);
      vl->setWidth(views[itView]->getWidth() / 2);
      vl->setHeight(views[itView]->getHeight());
      vl->setIntrinsicId(0);
      sfmData.getViews()[vl->getViewId()] = vl;

      std::shared_ptr<sfmData::View> vr = std::make_shared<sfmData::View>(v);
      vr->setViewId(itView * 2 + 1);
      vr->setRigAndSubPoseId(0, 1);
      vr->setImagePath(pathOutputs[0]);
      vr->setWidth(views[itView]->getWidth() / 2);
      vr->setHeight(views[itView]->getHeight());
      vr->setIntrinsicId(1);
      sfmData.getViews()[vr->getViewId()] = vr;
    }
  }

  if (!everyThingOK)
  {
    ALICEVISION_LOG_ERROR("Something wrent wrong while splitting");
    return EXIT_FAILURE;
  }

  for (auto id : toErase)
  {
    sfmData.getViews().erase(id);
  }

  // Export the SfMData scene in the expected format
  if(!sfmDataIO::Save(sfmData, outputSfmDataPath, sfmDataIO::ESfMData::ALL))
  {
    ALICEVISION_LOG_ERROR("An error occurred while trying to save '" << outputSfmDataPath << "'");
    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}
