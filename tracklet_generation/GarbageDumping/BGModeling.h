#ifndef _PROB_MODEL_H_
#define _PROB_MODEL_H_


/************************************************************************/
/* Basic Includes                                                       */
/************************************************************************/
#include	<iostream>
#include	<cstdlib>
#include	<cstring>
#include	<vector>
#include	<algorithm>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

/************************************************************************/
/*  Necessary includes for this Algorithm                               */
/************************************************************************/

#include "params.h"
using namespace cv;
#define WARP_MIX
#define AUTO_AE
//#define LEARN_SKIP
#define VAR_MIX_METHOD (1)


class CProbModel 
{

public:

	Mat	m_Cur;
	float		*m_DistImg;


	// 170623. Mat type conversion and Color image. 
	Mat Mean_[2];
	Mat Var_[2];
	Mat Age_[2];

	Mat Mean_tmp[2];
	Mat Var_tmp[2];
	Mat Age_tmp[2];

	Mat ModelIdx;


	// For ViBe mode
	Mat m_Samples[20];
	Mat m_Samples_temp[20];

	int	*m_ModelIdx;

	int	modelWidth;
	int modelHeight;

	int obsWidth;
	int obsHeight;

	// For Rapid illumination Change Check
	float m_bgMean;
	Mat mean_Mat;
	int nMargin;

public:
	CProbModel(void);
	~CProbModel(void);
	void uninit(void);
	void init(Mat pInputImg, int MCD_MODE);
	void motionCompensate(float h[9], int frame_num, float s, bool bInit);
	void update(Mat pOutputImg, Mat ROI, int frame_num, float s, bool bInit);
	void update_vibe(Mat pOutputImg, Mat AGE, Mat FGS, int nFrame, float fZero_ratio);
	void run(Mat curFrame, bool bMoving);

};

#endif // _PROB_MODEL_H_