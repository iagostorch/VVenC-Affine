/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   storchmain.h
 * Author: iagostorch
 *
 * Created on 5 de Janeiro de 2025, 12:48
 */

#ifndef STORCHMAIN_H
#define STORCHMAIN_H 1

#include "Unit.h"
#include <time.h>
#include <sys/time.h>


// Allows an easier handling of affine block sizes with less conditionals
typedef enum{
    _128x128 	= 0,
    _128x64     = 1,
    _64x128 	= 2,
    _64x64 	= 3,
    _64x32 	= 4,
    _32x64 	= 5,
    _64x16 	= 6,
    _16x64 	= 7,
    _32x32 	= 8,
    _32x16 	= 9,
    _16x32 	= 10,
    _16x16 	= 11,
    NUM_SIZES	= 12
} CuSize;

class storch {
    
public:
    
    // Custom encoding parameters
    static int sTRACE_xCompressCU, sTRACE_xPredAffineInterSearch;
    
    static CuSize getSizeEnum(vvenc::CodingUnit cu);
    
    
    static void startxPredAffineInterInterSearchUnipred_size( );
    static void finishxPredAffineInterInterSearchUnipred_size( );
    static void startxPredAffineInterInterSearch_size( );
    static void finishxPredAffineInterInterSearch_size( );
    static void printSummary();
    static bool isAffineSize(vvenc::SizeType width, vvenc::SizeType height);
    
    
    storch();
    
#if EXAMPLE || EXAMPLE
    static void exampleFunct();
#endif
    
private:    
    
    static int priv;
    
    static double xPredAffineInterSearch_time, xPredAffineInterSearchUnipred_time;
    static struct timeval xPredAffineInterSearch_tv1, xPredAffineInterSearch_tv2, xPredAffineInterSearchUnipred_tv1, xPredAffineInterSearchUnipred_tv2;
            
};


#endif /* STORCHMAIN_H */
