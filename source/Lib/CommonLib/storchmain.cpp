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

// Custom encoding parameters
int storch::sTRACE_xCompressCU, storch::sTRACE_xPredAffineInterSearch;

int storch::priv;
struct timeval storch::xPredAffineInterSearchUnipred_tv1, storch::xPredAffineInterSearchUnipred_tv2, storch::xPredAffineInterSearch_tv1, storch::xPredAffineInterSearch_tv2;
double storch::xPredAffineInterSearchUnipred_time, storch::xPredAffineInterSearch_time;



storch::storch(){
  priv = 2;
  xPredAffineInterSearchUnipred_time = 0.0;
  xPredAffineInterSearch_time = 0.0;
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