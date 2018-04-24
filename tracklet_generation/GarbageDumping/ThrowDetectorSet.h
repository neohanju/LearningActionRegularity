
/************************************************************************/
/* Basic Includes                                                       */
/************************************************************************/
#pragma once

#include <vector>
#include <queue>
#include <list>


#include	<iostream>
#include	<cstdlib>
#include	<cstring>
#include	<vector>
#include	<algorithm>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "haanju_utils.hpp"

#include "ThrowDetector.h"



/************************************************************************/
/*  Necessary includes for this Algorithm                               */
/************************************************************************/

#include "params.h"
using namespace cv;



/////////////////////////////////////////////////////////////////////////
// POSE CLASSIFICATION RESULT (OF ENTIRE TARGETS)
/////////////////////////////////////////////////////////////////////////

class CThrownResultSet
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CThrownResultSet() : frameIdx(0), timeStamp(0), procTime(0) {}
	~CThrownResultSet() {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int frameIdx;
	unsigned int timeStamp;
	time_t procTime;
	std::vector<CThrowDetector> throwResults;
};




class CThrowDetectorSet
{
public:

	// Image realted Mat
	Mat m_curFrame;
	Mat m_fore_up, m_heatmap_up;
	int imageWidth;
	int imageHeight;


	// Road region estimation...
	Mat m_fg_accum;
	Mat m_heat_accum;
	Mat m_mov_ped_accum;
	Mat m_mov_ped_accum_tmp;
	Mat m_prev_mov_ped;


	// For visualize matrix...
	Mat m_DispMat;


	unsigned int frameIdx;
	unsigned int timeStamp;
	time_t procTime;
	std::vector<CThrowDetector> listThrowResult;
	CThrownResultSet throwResult_;


public:
	CThrowDetectorSet(void);
	~CThrowDetectorSet(void);
	void uninit(void);
	void init(int width, int height);
	void Run(hj::CTrackResult trackResult, Mat matCurFrame, Mat m_fg_map, Mat Heatmap, Mat matDisp);
	void region_accumulate(Mat mat_fg, Mat mat_heat);

};


