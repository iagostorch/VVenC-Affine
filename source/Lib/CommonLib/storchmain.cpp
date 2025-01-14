/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   storchmain.cpp
 * Author: iagostorch
 * 
 * Created on 5 de Janeiro de 2025, 12:48
 */

#include <cstdio>
#include "storchmain.h"
#include "CodingStructure.h"
#include "Picture.h"

// Custom encoding parameters
int storch::sTRACE_xCompressCU, storch::sTRACE_xPredAffineInterSearch;
int storch::sEXTRACT_ameProgress, storch::sEXTRACT_frame;
int storch::sGPU_gpuMe2Cps, storch::sGPU_gpuMe3Cps, storch::sGPU_predict3CpsFrom2Cps;

// Variables 
std::ofstream storch::affine_me_2cps_file, storch::affine_me_3cps_file;
int storch::currPoc;
int storch::extractedFrames[EXT_NUM][500]; // Marks what frames were already extracted   

char storch::fillerChar;

int storch::priv;
struct timeval storch::xPredAffineInterSearchUnipred_tv1, storch::xPredAffineInterSearchUnipred_tv2, storch::xPredAffineInterSearch_tv1, storch::xPredAffineInterSearch_tv2;
double storch::xPredAffineInterSearchUnipred_time, storch::xPredAffineInterSearch_time;



storch::storch(){
  priv = 2;
  xPredAffineInterSearchUnipred_time = 0.0;
  xPredAffineInterSearch_time = 0.0;
  
  if(sEXTRACT_ameProgress)
  {
        affine_me_2cps_file.open("affine_progress_2CPs.csv");
        affine_me_3cps_file.open("affine_progress_3CPs.csv");
        
        
        affine_me_2cps_file << "POC,List,Ref,X,Y,W,H,";
        affine_me_2cps_file << "AMVP_0_X,AMVP_0_Y,AMVP_1_X,AMVP_1_Y,AMVP_2_X,AMVP_2_Y,";
        affine_me_2cps_file << "INIT_0_X,INIT_0_Y,INIT_1_X,INIT_1_Y,INIT_2_X,INIT_2_Y,";
        affine_me_2cps_file << "isGradient,";
        affine_me_2cps_file << " GRAD_0_X,GRAD_0_Y,GRAD_1_X,GRAD_1_Y,GRAD_2_X,GRAD_2_Y,";
        affine_me_2cps_file << "isRefinement,";
        affine_me_2cps_file << "FINAL_0_X,FINAL_0_Y,FINAL_1_X,FINAL_1_Y,FINAL_2_X,FINAL_2_Y" << std::endl;
        
        affine_me_3cps_file << "POC,List,Ref,X,Y,W,H,";
        affine_me_3cps_file << "AMVP_0_X,AMVP_0_Y,AMVP_1_X,AMVP_1_Y,AMVP_2_X,AMVP_2_Y,";
        affine_me_3cps_file << "INIT_0_X,INIT_0_Y,INIT_1_X,INIT_1_Y,INIT_2_X,INIT_2_Y,";
        affine_me_3cps_file << "isGradient,";
        affine_me_3cps_file << " GRAD_0_X,GRAD_0_Y,GRAD_1_X,GRAD_1_Y,GRAD_2_X,GRAD_2_Y,";
        affine_me_3cps_file << "isRefinement,";
        affine_me_3cps_file << "FINAL_0_X,FINAL_0_Y,FINAL_1_X,FINAL_1_Y,FINAL_2_X,FINAL_2_Y" << std::endl;
        
        fillerChar = 'x';
    }
  
    for(int f=0; f<500; f++){
      for(int t=0; t<EXT_NUM; t++){
          extractedFrames[t][f]=0;   // at the start, no frame was extracted
      }
    }
}

// Allows an easier handling of block sizes with meaninful names and less conditionals
CuSize storch::getSizeEnum(vvenc::CodingUnit cu){
  if(cu.lwidth()==128 && cu.lheight()==128)
    return _128x128;
  else if (cu.lwidth()==128 && cu.lheight()==64)
    return _128x64;
  else if (cu.lwidth()==64 && cu.lheight()==128)
    return _64x128;
  else if (cu.lwidth()==64 && cu.lheight()==64)
    return _64x64;
  else if (cu.lwidth()==64 && cu.lheight()==32)
    return _64x32;
  else if (cu.lwidth()==32 && cu.lheight()==64)
    return _32x64;
  else if (cu.lwidth()==64 && cu.lheight()==16)
    return _64x16;
  else if (cu.lwidth()==16 && cu.lheight()==64)
    return _16x64;
  else if (cu.lwidth()==32 && cu.lheight()==32)
    return _32x32;
  else if (cu.lwidth()==32 && cu.lheight()==16)
    return _32x16;
  else if (cu.lwidth()==16 && cu.lheight()==32)
    return _16x32;
  else if (cu.lwidth()==16 && cu.lheight()==16)
    return _16x16;
  else{
    printf("ERROR - Wrong PU size in getSizeEnum\n");
    exit(0);
  }
}

void storch::printSummary(){
  printf("Custom statistics:\n\n");
  printf("xPredAffineInterSearch: %f\n", storch::xPredAffineInterSearch_time);
  printf("xPredAffineInterSearch_unipred: %f\n", storch::xPredAffineInterSearchUnipred_time);
}

// Computation performed by the CPU for xPredAffineInterInterSearch method considering only the uniprediction stage
void storch::startxPredAffineInterInterSearchUnipred_size( ){
  gettimeofday(&xPredAffineInterSearchUnipred_tv1, NULL);
}
void storch::finishxPredAffineInterInterSearchUnipred_size( ){
  gettimeofday(&xPredAffineInterSearchUnipred_tv2, NULL);
  storch::xPredAffineInterSearchUnipred_time += (double) (storch::xPredAffineInterSearchUnipred_tv2.tv_usec - storch::xPredAffineInterSearchUnipred_tv1.tv_usec)/1000000 + (double) (storch::xPredAffineInterSearchUnipred_tv2.tv_sec - storch::xPredAffineInterSearchUnipred_tv1.tv_sec);
}

// Computation performed by the CPU for xPredAffineInterInterSearch method
void storch::startxPredAffineInterInterSearch_size( ){
  gettimeofday(&xPredAffineInterSearch_tv1, NULL);
}
void storch::finishxPredAffineInterInterSearch_size( ){
  gettimeofday(&xPredAffineInterSearch_tv2, NULL);
  storch::xPredAffineInterSearch_time += (double) (storch::xPredAffineInterSearch_tv2.tv_usec - storch::xPredAffineInterSearch_tv1.tv_usec)/1000000 + (double) (storch::xPredAffineInterSearch_tv2.tv_sec - storch::xPredAffineInterSearch_tv1.tv_sec);
}

bool storch::isAffineSize(vvenc::SizeType width, vvenc::SizeType height){
  if((width>=32) && (height>=32))
    return true;
  else if((width==32) && (height==16))
    return true;
  else if((width==16) && (height==32))
    return true;
  else if((width==64) && (height==16))
    return true;
  else if((width==16) && (height==64))
    return true;
  else if((width==16) && (height==16))
    return true;
  else
    return false;
}

void storch::exportAmeProgressFlag(int is3CPs, int flag){
    if(!is3CPs){ // 2 CPs
        storch::affine_me_2cps_file << flag << ",";
    }
    else{
        storch::affine_me_3cps_file << flag << ",";
    }
}

void storch::exportAmeProgressMVs(int is3CPs, vvenc::Mv mvs[3], int isFiller, int isFinal){
    if(!is3CPs){ // 2 CPs
        if(isFiller)
            storch::affine_me_2cps_file << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar;
        else
            storch::affine_me_2cps_file << mvs[0].hor << "," << mvs[0].ver << "," << mvs[1].hor << "," << mvs[1].ver << "," << mvs[2].hor << "," << mvs[2].ver;
        if(isFinal)
            storch::affine_me_2cps_file << std::endl;
        else
            storch::affine_me_2cps_file << ",";
    }
    else{
        if(isFiller)
            storch::affine_me_3cps_file << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar << "," << storch::fillerChar;
        else
            storch::affine_me_3cps_file << mvs[0].hor << "," << mvs[0].ver << "," << mvs[1].hor << "," << mvs[1].ver << "," << mvs[2].hor << "," << mvs[2].ver;
        
        if(isFinal)
            storch::affine_me_3cps_file << std::endl;
        else
            storch::affine_me_3cps_file << ",";
    }
}

void storch::exportAmeProgressBlock(int is3CPs, int refList, int refIdx, vvenc::CodingUnit& cu){
    if(!is3CPs){ // 2CPs
        storch::affine_me_2cps_file << cu.cs->picture->getPOC() << "," << refList << "," << refIdx << "," << cu.lx() << "," << cu.ly() << "," << cu.lwidth() << "," << cu.lheight() << ",";     
    }
    else{
        storch::affine_me_3cps_file << cu.cs->picture->getPOC() << "," << refList << "," << refIdx << "," << cu.lx() << "," << cu.ly() << "," << cu.lwidth() << "," << cu.lheight() << ",";
    }
}

// Export the samples of a frame into a CSV file
void storch::exportSamplesFrame(vvenc::CPelBuf samples, int POC, SamplesType type){
    int h,w;
    if(type == EXT_ORIGINAL){
        if(extractedFrames[EXT_ORIGINAL][POC] == 0){ // If the original frame was not extracted yet...
            std::ofstream fileHandler;

            std::string name = (std::string) "original_" + std::to_string(POC);

            fileHandler.open(name + ".csv");

            int frameWidth = samples.width;
            int frameHeight = samples.height;

            for (h=0; h<frameHeight; h++){
                for(w=0; w<frameWidth-1; w++){
                    fileHandler << samples.at(w,h) << ",";
                }
                fileHandler << samples.at(w,h);
                fileHandler << std::endl;
            }
            fileHandler.close();
            extractedFrames[EXT_ORIGINAL][POC] = 1; // Mark the current frame as extracted
        }
        else{
            std::cout << "ERROR - EXTRACTING THE SAME ORIGINAL FRAME TWICE" << std::endl;
        }        
    }
    else if(type == EXT_RECONSTRUCTED){
        if(extractedFrames[EXT_RECONSTRUCTED][POC] == 0){ // If the reconstructed frame was not extracted yet...
            std::ofstream fileHandler;

            std::string name = (std::string) "reconstructed_" + std::to_string(POC);

            fileHandler.open(name + ".csv");

            int frameWidth = samples.width;
            int frameHeight = samples.height;

            for (h=0; h<frameHeight; h++){
                for(w=0; w<frameWidth-1; w++){
                    fileHandler << samples.at(w,h) << ",";
                }
                fileHandler << samples.at(w,h);
                fileHandler << std::endl;
            }
            fileHandler.close();
            extractedFrames[EXT_RECONSTRUCTED][POC] = 1; // Mark the current frame as extracted
        }
    }
    else{
        std::cout << "ERROR - Incorrect type passes when exporting frame samples" << std::endl;
    }
}
