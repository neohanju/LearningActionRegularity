
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

#include "kcftracker.hpp"


/************************************************************************/
/*  Necessary includes for this Algorithm                               */
/************************************************************************/

#include "params.h"
using namespace cv;



/////////////////////////////////////////////////////////////////////////
// POSE CLASSIFICATION RESULT (OF ENTIRE TARGETS)
/////////////////////////////////////////////////////////////////////////



class CThrowDetector 
{

public: 
	// ID.
	unsigned int trackId;
	
	// State
	bool bCarrying;


	// KCF 
	float m_fDist_mean[18];
	float m_fDist_std[18];

	// Share
	float m_fDist_mean_obs[18];
	float m_fDist_std_obs[18];

	// MASK
	float m_fDist_mean_mask[18];
	float m_fDist_std_mask[18];

	Point2d m_ROI_LT;
	Point2d m_ROI_RB;

	bool m_bROI = false;
	bool m_bRHand = true; 

	



	Mat m_curFrame;
	Mat m_fore_up, m_heatmap_up;
	

	// ID for multiple object
//	int m_ID_index;

	// Input Keypoints..
	float _patch_w;
	float _patch_h;
	float _LH_x, _LH_y, _LH_c;
	float _min_fore_area;
	float _arr_joint_x[18];
	float _arr_joint_y[18];
	float _arr_joint_c[18];


	// KCF Tracker.
	KCFTracker tracker;
	Rect KCF_result;
	bool m_bTrackInit;

	// Decision
	bool m_bROI_warning;
	bool m_bThw_warning;
	bool m_bThw_warning2;	
	bool m_bFirstStart;


	int imageWidth;
	int imageHeight;

	bool m_bKCFRectInit;
	Rect _KCF_rect_init;
	int m_nReCnt;

	// 171130. 
	Rect _L_rect;
	Rect _R_rect;
	Rect _union_rect;
	Rect _KCF_rect_L;
	Rect _KCF_rect_R;
	int _nHandLR;
	float _max_BGProb_L;
	float _max_BGProb_R;

	bool bTrackON;

	// For visualize matrix...
	Mat m_DispMat; // memory link...


	//// 171213. result related.
	//std::vector<stThrowResult> listThwResult;
	//CThrowResultSet throwResult_;



public:
	CThrowDetector(void);
	~CThrowDetector(void);
	void uninit(void);
	void init(int width, int height);
	void ReInit(void);
	bool run_proposal(Mat curFrame, Mat foreground, Mat Heatmap);
	bool run_decision(Rect track_box, bool bMode);

	void set_ROI(int LT_x, int LT_y, int RB_x, int RB_y);
	void set_LHand(void);

	void carryingObjectProposal(Rect hand_rect, bool bHand, bool bTrackON);
	
	void FindCarryingObject(Rect region_rect, int min_fore_area, float th_heat_m, bool bHand, bool bTrackON);
	
	float MeasureJointBgProb(Rect region_rect);
	bool KCF_ReInit(Rect track_box);

	void set_keypoints(hj::CTrackResult trackResult, int idx);
	void set_keypoints_obj(hj::CObjectInfo objectInfo);

	bool Detect(Mat matCurFrame, Mat m_fg_map, Mat Heatmap, Mat matDisp);


	void InitKCFRect();

//	void Run(hj::CTrackResult trackResult, Mat matCurFrame, Mat m_fg_map, Mat Heatmap, Mat matDisp);

};

