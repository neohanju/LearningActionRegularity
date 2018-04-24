//#include "stdafx.h"
#include "ThrowDetector.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



CThrowDetector::CThrowDetector(void)
{



}

CThrowDetector::~CThrowDetector()
{
	uninit();
}

void CThrowDetector::uninit(void)
{

}

void CThrowDetector::init(int width, int height)
{

	memset(m_fDist_mean, 0, sizeof(float) * 18);
	memset(m_fDist_std, 10, sizeof(float) * 18);
	
	memset(_arr_joint_x, 0, sizeof(float) * 18);
	memset(_arr_joint_y, 0, sizeof(float) * 18);
	memset(_arr_joint_c, 0, sizeof(float) * 18);



	m_ROI_LT = Point(0, 0);
	m_ROI_RB = Point(0, 0);
	m_bROI = false;
	m_bRHand = true;
	
	imageWidth = width;
	imageHeight = height;

	// KCF related...
	m_bKCFRectInit = false;
	m_bTrackInit = false;



	m_nReCnt = 0;

	// Tracking ID... 
	//m_ID = -1; // 171110.
	//m_ID_index = -1; 


	// Decision
	m_bROI_warning = false;
	m_bThw_warning = false;
	m_bThw_warning2 = false;
	m_bFirstStart = false;

	// 171201.
	_nHandLR = -1;
	memset(m_fDist_mean_mask, 0, sizeof(float) * 18);
	memset(m_fDist_std_mask, 10, sizeof(float) * 18);

	memset(m_fDist_mean_obs, 0, sizeof(float) * 18);
	memset(m_fDist_std_obs, 10, sizeof(float) * 18);

	// STATE
	bCarrying = false;


}
void CThrowDetector::ReInit(void)
{

	memset(m_fDist_mean, 0, sizeof(float) * 18);
	memset(m_fDist_std, 10, sizeof(float) * 18);


	// 171201. Re-initialize the learned distance / std.
	memset(m_fDist_mean_mask, 0, sizeof(float) * 18);
	memset(m_fDist_std_mask, 10, sizeof(float) * 18);
	m_nReCnt = 0;

}




bool CThrowDetector::run_proposal(Mat curFrame, Mat foreground_up, Mat Heatmap_up)
{
	
	
	m_bKCFRectInit = false;
	m_bKCFRectInit = false;
	_KCF_rect_init = Rect(0, 0, 0, 0);
	_KCF_rect_L = Rect(0, 0, 0, 0);
	_KCF_rect_R = Rect(0, 0, 0, 0);
	_max_BGProb_L = 0.0;
	_max_BGProb_R = 0.0;


	
	if (m_nReCnt > 100)
	{
		memset(m_fDist_mean, 0, sizeof(float) * 18);
		memset(m_fDist_std, 10, sizeof(float) * 18);


		// 171201. Re-initialize the learned distance / std.
		memset(m_fDist_mean_mask, 0, sizeof(float) * 18);
		memset(m_fDist_std_mask, 10, sizeof(float) * 18);
		m_nReCnt = 0;

	}



	m_curFrame = curFrame;
//	curFrame.copyTo(m_DispMat); // copy to member variable
 //	Visualize_ROI();
	m_fore_up = foreground_up;
	m_heatmap_up = Heatmap_up;

	//// Match the foreground / heatmap size to input image Size.
	//resize(foreground, m_fore_up, Size(imageWidth, imageHeight), 0, 0, INTER_CUBIC);
	//resize(Heatmap, m_heatmap_up, Size(imageWidth, imageHeight), 0, 0, INTER_CUBIC);

	// Extract Initial Track Region using Mask & Heatmap info.
	if (_nHandLR == -1) // when hand is not set.
	{
		if (_L_rect.area() > 0)
			carryingObjectProposal(_L_rect, 0, false);
		if (_R_rect.area() > 0)
			carryingObjectProposal(_R_rect, 1, false);
	}
	else if (_nHandLR == 0 && _L_rect.area() > 0)
		carryingObjectProposal(_L_rect, 0, bTrackON);

	else if (_nHandLR == 1 && _R_rect.area() > 0)
		carryingObjectProposal(_R_rect, 1, bTrackON);






	if (_max_BGProb_L > _max_BGProb_R && _max_BGProb_L > 0)
	{
		_KCF_rect_init = _KCF_rect_L;
		_nHandLR = 0;
	}
	else if (_max_BGProb_R > _max_BGProb_L && _max_BGProb_R > 0)
	{
		_KCF_rect_init = _KCF_rect_R;
		_nHandLR = 1;
	}

	if (_KCF_rect_init.area() > 0)
	{
		// Visualize... 
		Mat heat_ex = m_heatmap_up(_KCF_rect_init);

	/*	namedWindow("contour2", WINDOW_NORMAL);
		resizeWindow("contour2", _KCF_rect_init.width * 2, _KCF_rect_init.height * 2);
		cv::imshow("contour2", heat_ex);*/

		// Visualize... 		
		rectangle(m_DispMat, Point(_KCF_rect_init.x, _KCF_rect_init.y), Point(_KCF_rect_init.x + _KCF_rect_init.width, _KCF_rect_init.y + _KCF_rect_init.height), Scalar(0, 0, 255), 1, LINE_AA);
		m_bKCFRectInit = true;


	}





		 


	// Visualize... 
//	if (m_bKCFRectInit == true)
//		rectangle(m_DispMat, Point(m_KCF_rect_init.x, m_KCF_rect_init.y), Point(m_KCF_rect_init.x + m_KCF_rect_init.width, m_KCF_rect_init.y + m_KCF_rect_init.height), Scalar(0, 0, 255), 1, LINE_AA);

	if (m_bKCFRectInit == false)
		m_nReCnt++;



	return m_bKCFRectInit;
	

}








void CThrowDetector::carryingObjectProposal(Rect hand_rect, bool bHand, bool bTrackON)
{

	if (hand_rect.x + hand_rect.width < imageWidth && hand_rect.y + hand_rect.height < imageHeight && hand_rect.x > 0 && hand_rect.y > 0)
	{

		// Visualize: Initial Rectangle Drawing around Hand...
		rectangle(m_DispMat, Point(hand_rect.x, hand_rect.y), Point(hand_rect.x + hand_rect.width, hand_rect.y + hand_rect.height), Scalar(0, 255, 0), 1, LINE_AA);


		float heat_m = MeasureJointBgProb(hand_rect);

		if (heat_m > 0.5)
		{
			FindCarryingObject(hand_rect, _min_fore_area, 0.5, bHand, bTrackON); // 171108. Simply high threshold can solve the prob?? Tradeoff between missing event??
		}



	}
	



	


}



void CThrowDetector::FindCarryingObject(Rect region_rect, int min_fore_area, float th_heat_m, bool bHand, bool bTrackON)
{

	Mat fore_patch = m_fore_up(region_rect);

	double threshold = 200; // 0 - 255 binarize....
	Mat fg_map_bin;
	Mat heatmap_bin;

	compare(fore_patch, threshold, fg_map_bin, CMP_GT);

	// 171130. heatmap combine degrade the performance...

	Mat heat_ex = m_heatmap_up(region_rect);

//	namedWindow("heatpatch", WINDOW_NORMAL);
//	resizeWindow("heatpatch", fore_patch.cols * 2, fore_patch.rows * 2);
//	imshow("heatpatch", heat_ex);

	double heatmap_th = 200; // 0 - 255 binarize....
	compare(heat_ex, heatmap_th, heatmap_bin, CMP_GT);
	Mat mul_mat = fg_map_bin.mul(heatmap_bin);

	//imshow("FG mul Heatmap", mul_mat);


	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;

	// 
	findContours(mul_mat, contours, hierarchy, RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0)); // Error in debug mode...

	float max_BGProb = -1.0; // probabilty criteria
	//	float max_moment = -1.0; // area criteria
	int max_cc = -1;
	_max_BGProb_L = max_BGProb;
	_max_BGProb_R = max_BGProb;



	// get the moment
	std::vector<Moments> mu(contours.size()); // moments for each contour
	std::vector<Rect> boundRect(contours.size());
	std::vector<std::vector<Point> > contours_poly(contours.size());

	if (contours.size() != 0)
	{
		// contour display
		RNG rng(12345);
		Mat drawing;
		fore_patch.copyTo(drawing);
		bool bPincnt = false;

		int contour_num = contours.size();
		double tmp_moment = 0.0;
		float jBGprob_tmp = 0.0;

		// 171129. Find most proper region candidatates... 
		// if throwing is not detected, the largest target is candidate....
		for (int cc = 0; cc < contour_num; cc++)
		{
			tmp_moment = moments(contours[cc], false).m00;
			if (tmp_moment > min_fore_area)
			{
				approxPolyDP(Mat(contours[cc]), contours_poly[cc], 3, true);
				boundRect[cc] = boundingRect(Mat(contours_poly[cc]));
				// Caution! boundRect start from the Initial Rect...

				
				if (bTrackON == true || m_bThw_warning == false || m_bThw_warning == false)
				{
					if (tmp_moment > min_fore_area && tmp_moment > max_BGProb)
					{
						max_cc = cc;
						max_BGProb = tmp_moment;
					}
				}

				else if (bTrackON == false)
				{
					jBGprob_tmp = MeasureJointBgProb(Rect(region_rect.x + boundRect[cc].x, region_rect.y + boundRect[cc].y, boundRect[cc].width, boundRect[cc].height));


					if (jBGprob_tmp > max_BGProb && jBGprob_tmp > th_heat_m)
					{
						max_cc = cc;
						max_BGProb = jBGprob_tmp;

					}

				}





				// extract region 
//				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//				drawContours(drawing, contours_poly, cc, color, 2, 8, hierarchy, 0, Point());
//				rectangle(drawing, boundRect[cc].tl(), boundRect[cc].br(), color, 2, 8, 0);

//				namedWindow("contour", WINDOW_NORMAL);
//				resizeWindow("contour", fore_patch.cols * 2, fore_patch.rows * 2);
//				imshow("contour", drawing);




				//	std::cout << jBGprob_tmp << std::endl;
				//	if (bTrackON == false)
				//		cv::waitKey();

			}

		}

	}

	if (max_cc >= 0)
	{

		if (bHand == 0)
		{
			_KCF_rect_L = Rect(region_rect.x + boundRect[max_cc].x, region_rect.y + boundRect[max_cc].y, boundRect[max_cc].width, boundRect[max_cc].height);
			_max_BGProb_L = max_BGProb;

			if (bTrackON == true && MeasureJointBgProb(_KCF_rect_L) < 0.3)
				_max_BGProb_L = -1;



		}
		if (bHand == 1)
		{
			_KCF_rect_R = Rect(region_rect.x + boundRect[max_cc].x, region_rect.y + boundRect[max_cc].y, boundRect[max_cc].width, boundRect[max_cc].height);
			_max_BGProb_R = max_BGProb;

			if (bTrackON == true && MeasureJointBgProb(_KCF_rect_R) < 0.3)
				_max_BGProb_L = -1;
		}
	}
}

float CThrowDetector::MeasureJointBgProb(Rect region_rect)
{
	Mat heatmap_patch = m_heatmap_up(region_rect);

	// Visualize: 
//	cv::imshow("patch_heatmap", heatmap_patch);



	Mat heat_ex = m_heatmap_up(region_rect);
	double heatmap_th = 190; // 0 - 255 binarize....
	Mat heat_map_bin;
	compare(heat_ex, heatmap_th, heat_map_bin, CMP_GT);
//	cv::imshow("patch_heatmap_bin", heat_map_bin);

	return mean(heat_map_bin / 255).val[0];
}

bool CThrowDetector::run_decision(Rect track_box, bool bMode)
{


	bool bWriteTxt = false;
	// Observation...
//	float m_fDist_mean_obs[18];
//	float m_fDist_std_obs[18];

	memset(m_fDist_mean_obs, 0, sizeof(float) * 18);
	memset(m_fDist_std_obs, 10, sizeof(float) * 18);

	
	// All joint distance keep and Decision is done by voting. 
	int nTotalJoint = 0;
	int nDesicionCnt = 0;

	// TEST. 
	for (int jj = 0; jj < 18; jj++)
	{

		float joint_x = _arr_joint_x[jj];
		float joint_y = _arr_joint_y[jj];
		float joint_c = _arr_joint_c[jj];


		if (joint_c > 0.3) // confidence 0.3? 0.5? 
		{


			float KCF_cx = track_box.x + (track_box.width) / 2;
			float KCF_cy = track_box.y + (track_box.height) / 2;


			line(m_DispMat, Point(KCF_cx, KCF_cy), Point(joint_x, joint_y), Scalar(128, 255, 128));

			if (bMode == 0)
			{
				// KCF
				m_fDist_mean_obs[jj] = sqrt((joint_x - KCF_cx)*(joint_x - KCF_cx) + (joint_y - KCF_cy)*(joint_y - KCF_cy));
				m_fDist_std_obs[jj] = MAX(m_fDist_mean_obs[jj] - m_fDist_mean[jj], 0);

				//fprintf(fp, "%f %f %f %f %f ", KCF_cx, joint_x, KCF_cy, joint_y, m_fDist_mean_obs[jj]);

				// Init the Joint Configuration...
				if (m_fDist_mean[jj] == 0 && joint_c > 0.5)
				{
					m_fDist_mean[jj] = m_fDist_mean_obs[jj];
					m_fDist_std[jj] = m_fDist_mean_obs[jj] * 0.05;

					
				}
				// If initialized, 
				else if (m_fDist_mean[jj] != 0)
				{
					if (m_fDist_std_obs[jj] > m_fDist_std[jj] * 2)
					{
						nDesicionCnt++;
						
					}
					nTotalJoint++;
				}
			}
			else
			{
				// Mask
				m_fDist_mean_obs[jj] = sqrt((joint_x - KCF_cx)*(joint_x - KCF_cx) + (joint_y - KCF_cy)*(joint_y - KCF_cy));
				m_fDist_std_obs[jj] = MAX(m_fDist_mean_obs[jj] - m_fDist_mean_mask[jj], 0);

				// Init the Joint Configuration...
				if (m_fDist_mean_mask[jj] == 0 && joint_c > 0.5)
				{
					m_fDist_mean_mask[jj] = m_fDist_mean_obs[jj];
					m_fDist_std_mask[jj] = m_fDist_mean_obs[jj] * 0.05;
				}
				// If initialized, 
				else if (m_fDist_mean_mask[jj] != 0)
				{
					if (m_fDist_std_obs[jj] > m_fDist_std_mask[jj] * 2)
					{
						nDesicionCnt++;
	
					}

					nTotalJoint++;
				}

			}


		}
		
	}

	// Decision...
	//		std::cout << nDesicionCnt << ", " << nTotalJoint << std::endl;
	if (nDesicionCnt * 3 > nTotalJoint && nTotalJoint > 8) // set the minimum joint number.
	{
		//putText(matDispKCF, "Throwing Detected", Point(10, 100), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255))
		return true;
	}
	else
	{
		// Update
		for (int jj = 0; jj < 18; jj++)
		{

			if (bMode == 0)
			{
				if (m_fDist_mean[jj] != 0)
				{
					float learning_rate = 0.95;
					m_fDist_mean[jj] = m_fDist_mean[jj] * learning_rate + m_fDist_mean_obs[jj] * (1.0 - learning_rate); // To Do: 

					if (m_fDist_std_obs[jj] > 0)
						m_fDist_std[jj] = m_fDist_std[jj] * learning_rate + m_fDist_std_obs[jj] * (1.0 - learning_rate); // To Do: 
				}

			}
			else
			{
				if (m_fDist_mean_mask[jj] != 0)
				{
					float learning_rate = 0.95;
					m_fDist_mean_mask[jj] = m_fDist_mean_mask[jj] * learning_rate + m_fDist_mean_obs[jj] * (1.0 - learning_rate); // To Do: 

					if (m_fDist_std_obs[jj] > 0)
						m_fDist_std_mask[jj] = m_fDist_std_mask[jj] * learning_rate + m_fDist_std_obs[jj] * (1.0 - learning_rate); // To Do: 
				}

			}

		}

		return false;

	}

	return false;

}



bool CThrowDetector::KCF_ReInit(Rect track_box)
{
	float KCF_cx, KCF_cy, mean_obs;
	Rect overlap;
	Rect search_rect;

	if (_nHandLR == 0)
		search_rect = _L_rect;
	else if (_nHandLR == 1)
		search_rect = _R_rect;
	else
		return false;

	if (overlap.area() == 0)
	{
		float KCF_cx = track_box.x + (track_box.width) / 2;
		float KCF_cy = track_box.y + (track_box.height) / 2;

		float mean_obs = sqrt((search_rect.x - KCF_cx)*(search_rect.x - KCF_cx) + (search_rect.y - KCF_cy)*(search_rect.y - KCF_cy));

		if (mean_obs > search_rect.width * 2)
		{
			return true;
		}

	}
	return false;




}

void CThrowDetector::set_keypoints(hj::CTrackResult trackResult, int idx)
{

	

	_patch_w = trackResult.objectInfos.at(idx).box.width;
	_patch_h = trackResult.objectInfos.at(idx).box.height / 2;

	for (int i = 0; i < 18; i++)
	{
		_arr_joint_x[i] = trackResult.objectInfos.at(idx).keyPoint.at(i).x;
		_arr_joint_y[i] = trackResult.objectInfos.at(idx).keyPoint.at(i).y;
		_arr_joint_c[i] = trackResult.objectInfos.at(idx).keyPoint.at(i).confidence;


	}

	float x4, y4, c4, x7, y7, c7;
	x4 = _arr_joint_x[4]; y4 = _arr_joint_y[4]; c4 = _arr_joint_c[4];
	x7 = _arr_joint_x[7]; y7 = _arr_joint_y[7]; c7 = _arr_joint_c[7];

	if (c4 > 0.5)
		_R_rect = Rect(x4 - _patch_w / 5, y4, _patch_w, _patch_h);
	else
		_R_rect = Rect(0, 0, 0, 0);

	if (c7 > 0.5)
		_L_rect = Rect(x7 - _patch_w, y7, _patch_w, _patch_h);
	else
		_L_rect = Rect(0, 0, 0, 0);

	_min_fore_area = MAX(0.01 * _patch_w * _patch_h, 100);

}

void CThrowDetector::set_keypoints_obj(hj::CObjectInfo objectInfo)
{



	_patch_w = objectInfo.box.width;
	_patch_h = objectInfo.box.height / 2;

	for (int i = 0; i < 18; i++)
	{
		_arr_joint_x[i] = objectInfo.keyPoint.at(i).x;
		_arr_joint_y[i] = objectInfo.keyPoint.at(i).y;
		_arr_joint_c[i] = objectInfo.keyPoint.at(i).confidence;


	}

	float x4, y4, c4, x7, y7, c7;
	x4 = _arr_joint_x[4]; y4 = _arr_joint_y[4]; c4 = _arr_joint_c[4];
	x7 = _arr_joint_x[7]; y7 = _arr_joint_y[7]; c7 = _arr_joint_c[7];

	if (c4 > 0.5)
		_R_rect = Rect(x4 - _patch_w / 5, y4, _patch_w, _patch_h);
	else
		_R_rect = Rect(0, 0, 0, 0);

	if (c7 > 0.5)
		_L_rect = Rect(x7 - _patch_w, y7, _patch_w, _patch_h);
	else
		_L_rect = Rect(0, 0, 0, 0);

	_min_fore_area = MAX(0.01 * _patch_w * _patch_h, 100);

}



bool  CThrowDetector::Detect(Mat matCurFrame, Mat m_fg_map, Mat Heatmap, Mat matDisp)
{
 	m_DispMat = matDisp;

	if (m_bTrackInit == false)
	{
		bool bTrackPos = run_proposal(matCurFrame, m_fg_map, Heatmap);

		if (bTrackPos == true)
		{
			tracker.init(_KCF_rect_init, matCurFrame);
			m_bTrackInit = true;
			bCarrying = true;
		//	cv::waitKey();

		}
	}
	else if (m_bTrackInit == true)
	{
		// 171201. memory link...
		

		// 2. Track RUN. 
		KCF_result = tracker.update(matCurFrame, m_fg_map);

		// 2.2 Mask info...
		bool bTrackPos = run_proposal(matCurFrame, m_fg_map, Heatmap);


		// 3. Throwing Action Decision using Voting concept...
		m_bThw_warning = run_decision(KCF_result, 0);

		// 3.2. Throwing Action Decision using MASK.
		if (bTrackPos)
			m_bThw_warning2 = run_decision(_KCF_rect_init, 1);
		else
			m_bThw_warning2 = false;



		// 4. IF Track fail, Tracker re initialize...
		float peak_Val = tracker.mPeak_value; // from tracker...
		//std::cout << peak_Val << std::endl;
		if (peak_Val < 0.15)
		{
			m_bTrackInit = false;
			m_bThw_warning = false;
		}
		// KCF tracker far away with hand... 171102. Problem??
		if (KCF_ReInit(KCF_result))
		{
			m_bTrackInit = false;
			m_bThw_warning = false;
		}

		// KCF visualize...
		rectangle(m_DispMat, Point(KCF_result.x, KCF_result.y), Point(KCF_result.x + KCF_result.width, KCF_result.y + KCF_result.height), Scalar(0, 255, 255), 1, LINE_AA);
		rectangle(m_DispMat, Point(_KCF_rect_init.x, _KCF_rect_init.y), Point(_KCF_rect_init.x + _KCF_rect_init.width, _KCF_rect_init.y + _KCF_rect_init.height), Scalar(100, 100, 255), 1, LINE_AA);

	}


	// 5. Combine with Prior ROI.
	// Human & ROI intersectiong exist, the alarm occurs...
	// This module will move to another class...

//	if (m_bROI)
//	{
//	
////		trackResult.objectInfos.at(idx).box.width;
//		Rect person_rect = Rect(trackResult.objectInfos.at(idx).box.x, trackResult.objectInfos.at(idx).box.y, trackResult.objectInfos.at(idx).box.width, trackResult.objectInfos.at(idx).box.height);
//		Rect overlap_ROI_person = Rect(m_ROI_LT.x, m_ROI_LT.y, m_ROI_RB.x - m_ROI_LT.x, m_ROI_RB.y - m_ROI_LT.y) & person_rect;
//
//		if (overlap_ROI_person.area() > 20)
//		{
//			m_bROI_warning = true;
//		}
//		else
//			m_bROI_warning = false;
//
//		// 5.2. Tracker disappear in the ROI, tkhe occlusion message appears. (ressonable?)
//	}
//	else
		m_bROI_warning = false;

	// EVENT VISUALIZE....


	// 6. Road Region Estimation using Attenuated Foreground map...
	// moving average by foreground map.
	//	GaussianBlur(m_fg_map, m_fg_map, Size(10, 10), 0, 0);
		
		// 171201. Not use yet.  
		//region_accumulate(m_fg_map, Heatmap);



	if (m_bThw_warning)
		putText(m_DispMat, "Throwing Detected (KCF)", Point(10, 100), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));

	if (m_bThw_warning2)
		putText(m_DispMat, "Throwing Detected (MASK)", Point(10, 150), CV_FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));




	return true;



}
void CThrowDetector::InitKCFRect()
{
	if (_max_BGProb_L > _max_BGProb_R && _max_BGProb_L > 0)
	{
		_KCF_rect_init = _KCF_rect_L;
		_nHandLR = 0;
	}
	else if (_max_BGProb_R > _max_BGProb_L && _max_BGProb_R > 0)
	{
		_KCF_rect_init = _KCF_rect_R;
		_nHandLR = 1;
	}

	if (_KCF_rect_init.area() > 0)
	{
		// Visualize... 
		//Mat heat_ex = m_heatmap_up(_KCF_rect_init);

		//namedWindow("contour2", WINDOW_NORMAL);
		//resizeWindow("contour2", _KCF_rect_init.width * 2, _KCF_rect_init.height * 2);
		//cv::imshow("contour2", heat_ex);

		// Visualize... 		
		rectangle(m_DispMat, Point(_KCF_rect_init.x, _KCF_rect_init.y), Point(_KCF_rect_init.x + _KCF_rect_init.width, _KCF_rect_init.y + _KCF_rect_init.height), Scalar(0, 0, 255), 1, LINE_AA);
		m_bKCFRectInit = true;


	}






}

