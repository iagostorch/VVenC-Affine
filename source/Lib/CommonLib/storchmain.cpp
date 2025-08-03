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
int storch::sGPU_gpuMe2Cps, storch::sGPU_gpuMe3Cps, storch::sGPU_predict3CpsFrom2Cps, storch::sGPU_extraGradientIterations;

// Variables 
std::ofstream storch::affine_me_2cps_file, storch::affine_me_3cps_file;
int storch::currPoc;
int storch::extractedFrames[EXT_NUM][500]; // Marks what frames were already extracted   

char storch::fillerChar;

int storch::priv;
struct timeval storch::xPredAffineInterSearchUnipred_tv1, storch::xPredAffineInterSearchUnipred_tv2, storch::xPredAffineInterSearch_tv1, storch::xPredAffineInterSearch_tv2;
double storch::xPredAffineInterSearchUnipred_time, storch::xPredAffineInterSearch_time;

double storch::gpuAme_time, storch::gpuNonAmeUseful_time, storch::gpuNonAmeUseless_time, storch::gpuNonAmeVariableCreation_time, storch::gpuNonAmeOthers_time, storch::gpuNonAmeFinalizing_time;

std::unordered_map<__pid_t, struct timeval> storch::hashmap_tv_affineMe;
std::unordered_map<__pid_t, double> storch::gpuAme_time_multithread;
struct timeval storch::gpuAme_tv1, storch::gpuAme_tv2, storch::gpuNonAmeUseful_tv1, storch::gpuNonAmeUseful_tv2, storch::gpuNonAmeUseless_tv1, storch::gpuNonAmeUseless_tv2;
struct timeval storch::gpuNonAmeVariableCreation_tv1, storch::gpuNonAmeVariableCreation_tv2, storch::gpuNonAmeOthers_tv1, storch::gpuNonAmeOthers_tv2, storch::gpuNonAmeFinalizing_tv1, storch::gpuNonAmeFinalizing_tv2;


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
  
  xPredAffineInterSearchUnipred_time = 0.0;
  xPredAffineInterSearch_time = 0.0;
  gpuAme_time = 0.0;
  gpuNonAmeUseful_time = 0.0;
  gpuNonAmeUseless_time = 0.0;
  gpuNonAmeVariableCreation_time = 0.0;
  gpuNonAmeOthers_time = 0.0;
  gpuNonAmeFinalizing_time = 0.0;
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

void storch::printCustomParams()
{
  printf("\n\n");
  printf("Customized input parameters:\n");
  printf("sGPU_gpuMe2Cps:               %d\n", storch::sGPU_gpuMe2Cps);
  printf("sGPU_gpuMe3Cps:               %d\n", storch::sGPU_gpuMe3Cps);
  printf("sGPU_predict3CpsFrom2Cps:     %d\n", storch::sGPU_predict3CpsFrom2Cps);
  printf("sGPU_extraGradientIterations: %d\n", storch::sGPU_extraGradientIterations);
  printf("\n\n");
}

void storch::printSummary(){
  printf("Custom statistics:\n\n");  
  printf("xPredAffineInterSearch:         %f\n", xPredAffineInterSearch_time);
  printf("xPredAffineInterSearch_unipred: %f\n", xPredAffineInterSearchUnipred_time);
  printf("Non-AME Useless time:           %f\n", gpuNonAmeUseless_time);
  printf("Non-AME Useful time:            %f\n", gpuNonAmeUseful_time);
  printf("Variable creation time:         %f\n", gpuNonAmeVariableCreation_time);
  printf("Others time:                    %f\n", gpuNonAmeOthers_time);
  printf("Finalizing time:                %f\n", gpuNonAmeFinalizing_time);
  printf("AME time:                       %f\n", gpuAme_time);
  printf("Individual time for %ld threads\n", gpuAme_time_multithread.size());
  double acc = 0.0;
  for(auto& it : gpuAme_time_multithread){
    printf("    AME Thread %d: %f\n", it.first, it.second);
    acc += it.second;
  }
  printf("    AME Acc: %f\n", acc);
  
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

// Computation that is performed by the GPU. We must substitute this measure by the GPU running time
void storch::startGpuPartAffineME_size(){
  gettimeofday(&gpuAme_tv1, NULL);
  
  auto tid = gettid();
  if(storch::hashmap_tv_affineMe.find(tid) == storch::hashmap_tv_affineMe.end()){ // First time. Create tv object
    struct timeval first_tv;
    gettimeofday(&first_tv, NULL);
    double dummy; // Used to initialize the time counter
    storch::hashmap_tv_affineMe.insert({tid, first_tv});
    
  }
  else{ // Only update tv
    struct timeval first_tv; 
    gettimeofday(&first_tv, NULL);
    storch::hashmap_tv_affineMe.at(tid) = first_tv;
  }
}

void storch::finishGpuPartAffineME_size(){
  gettimeofday(&gpuAme_tv2, NULL);
  double deltaT = (double) (storch::gpuAme_tv2.tv_usec - storch::gpuAme_tv1.tv_usec)/1000000 + (double) (storch::gpuAme_tv2.tv_sec - storch::gpuAme_tv1.tv_sec);
  
  storch::gpuAme_time += deltaT;

  // AME Time for each thread
  
  auto tid = gettid();
  
  struct timeval second_tv;
  gettimeofday(&second_tv, NULL);
  struct timeval first_tv = storch::hashmap_tv_affineMe.at(tid);

  deltaT = (double) (second_tv.tv_usec - first_tv.tv_usec)/1000000 + (double) (second_tv.tv_sec - first_tv.tv_sec);

  if(storch::gpuAme_time_multithread.find(tid) == storch::gpuAme_time_multithread.end()){ // First time. Create time object
    storch::gpuAme_time_multithread.insert({tid, deltaT});
  }
  else{
    storch::gpuAme_time_multithread.at(tid) += deltaT;
  }

  
}
// Computation performed by the CPU (inside xPredAffineInterInterSearch) that MUST be done even with GPU acceleration
void storch::startNonGpuUsefulAffineME_size(){
  gettimeofday(&gpuNonAmeUseful_tv1, NULL);
}
void storch::finishNonGpuUsefulAffineME_size(){
  gettimeofday(&gpuNonAmeUseful_tv2, NULL);
  storch::gpuNonAmeUseful_time += (double) (storch::gpuNonAmeUseful_tv2.tv_usec - storch::gpuNonAmeUseful_tv1.tv_usec)/1000000 + (double) (storch::gpuNonAmeUseful_tv2.tv_sec - storch::gpuNonAmeUseful_tv1.tv_sec);
}
// Computation performed by the CPU (inside xPredAffineInterInterSearch) that is useless when using GPU acceleration
void storch::startNonGpuUselessAffineME_size(){
  gettimeofday(&gpuNonAmeUseless_tv1, NULL);
}
void storch::finishNonGpuUselessAffineME_size(){
  gettimeofday(&gpuNonAmeUseless_tv2, NULL);
  storch::gpuNonAmeUseless_time += (double) (storch::gpuNonAmeUseless_tv2.tv_usec - storch::gpuNonAmeUseless_tv1.tv_usec)/1000000 + (double) (storch::gpuNonAmeUseless_tv2.tv_sec - storch::gpuNonAmeUseless_tv1.tv_sec);
}

// Computation performed by the CPU for creating the variables for the xPredAffineInterInterSearch variables
void storch::startNonGpuVariableCreation_size(){
  gettimeofday(&gpuNonAmeVariableCreation_tv1, NULL);
}
void storch::finishNonGpuVariableCreation_size(){
  gettimeofday(&gpuNonAmeVariableCreation_tv2, NULL);
  storch::gpuNonAmeVariableCreation_time += (double) (storch::gpuNonAmeVariableCreation_tv2.tv_usec - storch::gpuNonAmeVariableCreation_tv1.tv_usec)/1000000 + (double) (storch::gpuNonAmeVariableCreation_tv2.tv_sec - storch::gpuNonAmeVariableCreation_tv1.tv_sec);
}

// Computation performed by the CPU for other tasks (not useless but maybe not that useful) in the xPredAffineInterInterSearch method
void storch::startNonGpuOthers_size(){
  gettimeofday(&gpuNonAmeOthers_tv1, NULL);
}
void storch::finishNonGpuOthers_size(){
  gettimeofday(&gpuNonAmeOthers_tv2, NULL);
  storch::gpuNonAmeOthers_time += (double) (storch::gpuNonAmeOthers_tv2.tv_usec - storch::gpuNonAmeOthers_tv1.tv_usec)/1000000 + (double) (storch::gpuNonAmeOthers_tv2.tv_sec - storch::gpuNonAmeOthers_tv1.tv_sec);
}

// Computation performed by the CPU for finalizing the uniprediciton inside xPredAffineInterInterSearch, i.e., setting the motion field and so on
void storch::startNonGpuFinalizing_size(){
  gettimeofday(&gpuNonAmeFinalizing_tv1, NULL);
}
void storch::finishNonGpuFinalizing_size(){
  gettimeofday(&gpuNonAmeFinalizing_tv2, NULL);
  storch::gpuNonAmeFinalizing_time += (double) (storch::gpuNonAmeFinalizing_tv2.tv_usec - storch::gpuNonAmeFinalizing_tv1.tv_usec)/1000000 + (double) (storch::gpuNonAmeFinalizing_tv2.tv_sec - storch::gpuNonAmeFinalizing_tv1.tv_sec);
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
