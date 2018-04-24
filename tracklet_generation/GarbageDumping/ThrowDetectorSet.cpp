//#include "stdafx.h"
#include "ThrowDetectorSet.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


CThrowDetectorSet::CThrowDetectorSet(void)
{



}

CThrowDetectorSet::~CThrowDetectorSet()
{
	uninit();
}

void CThrowDetectorSet::uninit(void)
{

}


void CThrowDetectorSet::init(int width, int height)
{
	imageWidth = width;
	imageHeight = height;
	listThrowResult.clear();
}


void CThrowDetectorSet::Run(hj::CTrackResult trackResult, Mat matCurFrame, Mat m_fg_map, Mat Heatmap, Mat matDisp)
{
	matCurFrame.copyTo(m_DispMat); // copy to member variable

	
	listThrowResult.clear();

	//// Match the foreground / heatmap size to input image Size.
	resize(m_fg_map, m_fore_up, Size(imageWidth, imageHeight), 0, 0, INTER_CUBIC);
	resize(Heatmap, m_heatmap_up, Size(imageWidth, imageHeight), 0, 0, INTER_CUBIC);

	// Iterator for Observed Person from TrackResult
	for (std::vector<hj::CObjectInfo>::iterator objIter = trackResult.objectInfos.begin();
		objIter != trackResult.objectInfos.end(); objIter++)
	{
		bool bMatch = false;

		// ID Matching with ThwResult
		for (std::vector<CThrowDetector>::iterator prevResultIter = throwResult_.throwResults.begin();
			prevResultIter != throwResult_.throwResults.end(); prevResultIter++)
		{
			if (prevResultIter->trackId != objIter->id) { continue; } // ID 다르면 그대로 가고, 


			// ID가 같을 때 판단하기...
			// set keypoints
			//prevResultIter->set_keypoints(trackResult, objIter->id);
			prevResultIter->set_keypoints_obj(*objIter);


			// Throwing action detection...
			bool bDetect = prevResultIter->Detect(matCurFrame, m_fore_up, m_heatmap_up, matDisp); // should be pointer?? 
			bMatch = true;


			// Change to 
			listThrowResult.push_back(*prevResultIter);
		}
	
		if (bMatch == false)
		{
			
			CThrowDetector curThwDetector;
			curThwDetector.init(imageWidth, imageHeight);
			curThwDetector.trackId = objIter->id;
			listThrowResult.push_back(curThwDetector);
		}

		
	}

	throwResult_.throwResults.clear();
	throwResult_.throwResults = listThrowResult;


}




void  CThrowDetectorSet::region_accumulate(Mat mat_fg, Mat mat_heat)
{
	float lr_decay = 1.0;

	// 171106. Maybe not used...
	// Foreground Accumulation Map... 
	//unsigned char* fg_data = (unsigned char*)(mat_fg.data);
	//unsigned char* fg_accum_data = (unsigned char*)(m_fg_accum.data);



	//// Caution: foreground map size and heatmap size are different...
	//if (m_fg_accum.cols <= 0)
	//	m_fg_accum = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
	//else
	//{
	//	addWeighted(m_fg_accum, lr_decay, mat_fg, lr_decay, 0, m_fg_accum);
	//	//for (int j = 0; j < mat_fg.rows; j++) 
	//	//{
	//	//	for (int i = 0; i < mat_fg.cols;i++) 
	//	//	{
	//	//		if (fg_data[i + j*(mat_fg.cols)] > 0)
	//	//		{
	//	//			//	fg_accum_data[i + j*(mat_fg.cols)] = (1.0 - lr_decay) * fg_accum_data[i + j*(mat_fg.cols)] + lr_decay * fg_data[i + j*(mat_fg.cols)];
	//	//			fg_accum_data[i + j*(mat_fg.cols)] = 255;
	//	//		}
	//	//		else
	//	//			fg_accum_data[i + j*(mat_fg.cols)] = lr_decay * fg_accum_data[i + j*(mat_fg.cols)];
	//	//	}
	//	//}


	//}


	// Heatmap Accumulation Map...
	//unsigned char* heat_data = (unsigned char*)(mat_heat.data);
	//unsigned char* heat_accum_data = (unsigned char*)(m_heat_accum.data);

	//if (m_heat_accum.cols <= 0)
	//{
	//	mat_heat.copyTo(m_heat_accum);
	//	m_heat_accum = 255 - m_heat_accum;
	//}
	//	
	//else
	//{
	//	addWeighted(m_heat_accum, lr_decay, 255 - mat_heat, lr_decay, 0, m_heat_accum);
	//	//for (int j = 0; j < mat_heat.rows; j++)
	//	//{
	//	//	for (int i = 0; i < mat_heat.cols; i++)
	//	//	{
	//	//		if (255 - heat_data[i + j*(mat_heat.cols)] > 128)
	//	//		{
	//	//			heat_accum_data[i + j*(mat_heat.cols)] = (1.0 - lr_decay) * heat_accum_data[i + j*(mat_heat.cols)] + lr_decay * (255 - heat_data[i + j*(mat_heat.cols)]);
	//	//			//heat_accum_data[i + j*(mat_heat.cols)] = 255;
	//	//		}
	//	//		else
	//	//			heat_accum_data[i + j*(mat_heat.cols)] = lr_decay * heat_accum_data[i + j*(mat_fg.cols)];
	//	//	}
	//	//}



	//}


	//cv::imshow("Road Region FG", m_fg_accum);
	//cv::imshow("Road Region Heat", m_heat_accum);


	// Moving X Pedestrian...
	Mat heat_th;
	resize(255 - mat_heat, heat_th, cv::Size(mat_fg.cols, mat_fg.rows), 0, 0, 1);
	threshold(heat_th, heat_th, 50, 255, 0); // binary... threshold function: if 0, 255 set, if 3, the value is preserved...
	//	cv::imshow("Heatmap threshold", heat_th);
	//	cv::imshow("FG inverted", 255 - mat_fg);



	Mat mov_ped;
	cv::multiply(mat_fg, heat_th, mov_ped, 1.0 / 255.0);



	if (m_mov_ped_accum.cols <= 0)
	{
		m_mov_ped_accum = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
		m_prev_mov_ped = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
		m_mov_ped_accum_tmp = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
		mov_ped = Mat::zeros(mat_fg.rows, mat_fg.cols, CV_8UC1);
	}
	else
	{
		//		addWeighted(m_mov_ped_accum, lr_decay, mov_ped, lr_decay, 0, m_mov_ped_accum);

		unsigned char* p_mov_ped = (unsigned char*)(mov_ped.data);
		unsigned char* p_m_mov_ped_accum = (unsigned char*)(m_mov_ped_accum.data);
		unsigned char* p_m_mov_ped_accum_tmp = (unsigned char*)(m_mov_ped_accum_tmp.data);
		unsigned char* p_m_prev_mov_ped = (unsigned char*)(m_prev_mov_ped.data);


		//		cv::imshow("Accumulumation Map2", mov_ped);
		//		cv::imshow("Accumulumation Map3", m_prev_mov_ped);

		for (int j = 0; j < mat_fg.rows; j++)
		{
			for (int i = 0; i < mat_fg.cols; i++)
			{
				if (p_mov_ped[i + j*(mat_fg.cols)] > 100 && p_m_prev_mov_ped[i + j*(mat_fg.cols)] > 100)
				{
					p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)]++; // Count the staying pixel by consecutive foreground...
					if (p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)] > p_m_mov_ped_accum[i + j*(mat_fg.cols)]) // Compare pre-stored staying pixels
					{
						p_m_mov_ped_accum[i + j*(mat_fg.cols)] = p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)]; // Swap the staying pixels...
					}

				}
				else
				{
					//if (p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)] > p_m_mov_ped_accum[i + j*(mat_fg.cols)]) // Compare pre-stored staying pixels
					//{
					//	p_m_mov_ped_accum[i + j*(mat_fg.cols)] = p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)]; // Swap the staying pixels...
					//}
					p_m_mov_ped_accum_tmp[i + j*(mat_fg.cols)] = 0; // re-initialized the count...
				}

			}
		}
	}



	//	cv::imshow("Accumulumation Map", m_mov_ped_accum);

	mov_ped.copyTo(m_prev_mov_ped);


	//	pAge_tmp[i + j*modelWidth] = MIN(pAge_tmp[i + j*modelWidth] * exp(-VAR_DEC_RATIO*MAX(0.0, pVar_tmp[i + j*modelWidth] - VAR_MIN_NOISE_T)), MAX_BG_AGE);


}