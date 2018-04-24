//#include "stdafx.h"
#include "BGModeling.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



CProbModel::CProbModel(void)
{

}

CProbModel::~CProbModel()
{
	uninit();
}

void CProbModel::uninit(void)
{

}

void CProbModel::init(Mat pInputImg, int MCD_MODE)
{
	uninit();

	m_Cur = pInputImg;
	

	obsWidth = pInputImg.cols;
	obsHeight = pInputImg.rows;

	modelWidth = obsWidth / BLOCK_SIZE_X;
	modelHeight = obsHeight / BLOCK_SIZE_Y;

//	mean_Mat = Mat::zeros(modelHeight, modelWidth, CV_8UC1);


	/////////////////////////////////////////////////////////////////////////////
	// Initialize Storage

	//		mean_Mat = Mat::zeros(modelHeight, modelWidth, CV_8UC1);
	for (int m = 0; m < 2; m++)
	{
		Mean_[m] = Mat::zeros(modelHeight, modelWidth, CV_32FC3);
		Var_[m] = Mat::zeros(modelHeight, modelWidth, CV_32FC1); // Variance is 1 or 3 channel? 
		Age_[m] = Mat::zeros(modelHeight, modelWidth, CV_32FC1); // age = 1 channel. 
	
		Mean_tmp[m] = Mat::zeros(modelHeight, modelWidth, CV_32FC3);
		Var_tmp[m] = Mat::zeros(modelHeight, modelWidth, CV_32FC1);
		Age_tmp[m] = Mat::zeros(modelHeight, modelWidth, CV_32FC1);

	}
	

	ModelIdx = Mat::zeros(modelHeight, modelWidth, CV_8UC1);


	// update with homography 
	float h[9];
	h[0] = 1.0;	h[1] = 0.0;	h[2] = 0.0;
	h[3] = 0.0;	h[4] = 1.0;	h[5] = 0.0;
	h[6] = 0.0;	h[7] = 0.0;	h[8] = 1.0;


//	motionCompensate(h, 1, 1, true);
	//		update(NULL, NULL);



	Scalar MM = mean(m_Cur);
	m_bgMean = MM.val[0];

	nMargin = 0;


	if (MCD_MODE == 1)
	{
		// ViBE samples initializer. 
		for (int j = 0; j < 20; j++)
		{
			m_Cur.copyTo(m_Samples[j]);
			m_Cur.copyTo(m_Samples_temp[j]);
		}
	}





}

void CProbModel::motionCompensate(float h[9], int frame_num, float s, bool bInit)
{

	int var_mix_method = VAR_MIX_METHOD;
	int skip_level;

#ifdef LEARN_SKIP

	if (bInit == true)
	{
		skip_level = 1;
		var_mix_method = 1;
	}
	else if (s < 2)
	{
		skip_level = 2;
		if (frame_num % skip_level == 0)
			var_mix_method = 3;
	}
#endif 


	int curModelWidth = modelWidth;
	int curModelHeight = modelHeight;


	unsigned char* pCur = (unsigned char*)(m_Cur.data);


	

	int obsWidthStep = m_Cur.cols;

	// compensate models for the current view
	for (int j = 0; j < curModelHeight; ++j){
		for (int i = 0; i < curModelWidth; ++i){

			// x and y coordinates for current model
			float X, Y;
			float W = 1.0;
			X = BLOCK_SIZE_X*i + BLOCK_SIZE_X / 2.0;
			Y = BLOCK_SIZE_Y*j + BLOCK_SIZE_Y / 2.0;

			// transformed coordinates with h
			float newW = h[6] * X + h[7] * Y + h[8];
			float newX = (h[0] * X + h[1] * Y + h[2]) / newW;
			float newY = (h[3] * X + h[4] * Y + h[5]) / newW;

			// transformed i,j coordinates of old position
			float newI = newX / BLOCK_SIZE_X;
			float newJ = newY / BLOCK_SIZE_Y;

			int idxNewI = floor(newI);
			int idxNewJ = floor(newJ);

			float di = newI - ((float)(idxNewI)+0.5);
			float dj = newJ - ((float)(idxNewJ)+0.5);

			float w_H = 0.0;
			float w_V = 0.0;
			float w_HV = 0.0;
			float w_self = 0.0;
			float sumW = 0.0;
			float sumW2 = 0.0;

//			int idxNow = i + j*modelWidth;


			// For Mean and Age
			{
				float temp_mean_R[4][NUM_MODELS];
				float temp_mean_G[4][NUM_MODELS];
				float temp_mean_B[4][NUM_MODELS];

				float temp_age[4][NUM_MODELS];
				memset(temp_mean_R, 0, sizeof(float)* 4 * NUM_MODELS);
				memset(temp_mean_G, 0, sizeof(float)* 4 * NUM_MODELS);
				memset(temp_mean_B, 0, sizeof(float)* 4 * NUM_MODELS);
				memset(temp_age, 0, sizeof(float)* 4 * NUM_MODELS);






#ifdef WARP_MIX
				// Horizontal Neighbor
				if (di != 0){
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight)
					{
						w_H = abs(di) * (1.0 - abs(dj));
						sumW += w_H;
//						3*idx_new_i + 3*idx_new_j*modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m)
						{

							float* pMean_ = (float*)(Mean_[m].data);
							temp_mean_R[0][m] = w_H * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth];
							temp_mean_G[0][m] = w_H * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth + 1];
							temp_mean_B[0][m] = w_H * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth + 2];

							float* pAge_ = (float*)(Age_[m].data);
							temp_age[0][m] = w_H * pAge_[idx_new_i + idx_new_j*modelWidth];
						}
					}
				}

				// Vertical Neighbor
				if (dj != 0){
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight)
					{
						w_V = abs(dj) * (1.0 - abs(di));
						sumW += w_V;
						int idxNew = idx_new_i + idx_new_j*modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m)
						{
				
							float* pMean_ = (float*)(Mean_[m].data);
							temp_mean_R[1][m] = w_V * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth];
							temp_mean_G[1][m] = w_V * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth + 1];
							temp_mean_B[1][m] = w_V * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth + 2];

							float* pAge_ = (float*)(Age_[m].data);
							temp_age[1][m] = w_V * pAge_[idx_new_i + idx_new_j*modelWidth];
						}
					}
				}


				// HV Neighbor
				if (dj != 0 && di != 0){
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight)
					{
						w_HV = abs(di) * abs(dj);
						sumW += w_HV;
						int idxNew = idx_new_i + idx_new_j*modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m)
						{
							float* pMean_ = (float*)(Mean_[m].data);
							temp_mean_R[2][m] = w_HV * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth];
							temp_mean_G[2][m] = w_HV * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth + 1];
							temp_mean_B[2][m] = w_HV * pMean_[3 * idx_new_i + 3 * idx_new_j*modelWidth + 2];

							float* pAge_ = (float*)(Age_[m].data);
							temp_age[2][m] = w_HV * pAge_[idx_new_i + idx_new_j*modelWidth];
						}
					}
				}
#endif
				// Self
				if (idxNewI >= 0 && idxNewI < curModelWidth && idxNewJ >= 0 && idxNewJ < curModelHeight){
					w_self = (1.0 - abs(di)) * (1.0 - abs(dj));
					sumW += w_self;
					int idxNew = idxNewI + idxNewJ*modelWidth;
					for (int m = 0; m < NUM_MODELS; ++m){
				

						float* pMean_ = (float*)(Mean_[m].data);
						temp_mean_R[3][m] = w_self * pMean_[3 * idxNewI + 3 * idxNewJ*modelWidth];
						temp_mean_G[3][m] = w_self * pMean_[3 * idxNewI + 3 * idxNewJ*modelWidth + 1];
						temp_mean_B[3][m] = w_self * pMean_[3 * idxNewI + 3 * idxNewJ*modelWidth + 2];

						float* pAge_ = (float*)(Age_[m].data);
						temp_age[3][m] = w_self * pAge_[idxNewI + idxNewJ*modelWidth];




					}
				}

				if (sumW > 0){
					for (int m = 0; m < NUM_MODELS; ++m){
#ifdef WARP_MIX

					
						float* pMean_tmp = (float*)(Mean_tmp[m].data);
						pMean_tmp[3 * i + j*modelWidth * 3] = (temp_mean_R[0][m] + temp_mean_R[1][m] + temp_mean_R[2][m] + temp_mean_R[3][m]) / sumW;
						pMean_tmp[3 * i + j*modelWidth * 3 + 1] = (temp_mean_G[0][m] + temp_mean_G[1][m] + temp_mean_G[2][m] + temp_mean_G[3][m]) / sumW;
						pMean_tmp[3 * i + j*modelWidth * 3 + 2] = (temp_mean_B[0][m] + temp_mean_B[1][m] + temp_mean_B[2][m] + temp_mean_B[3][m]) / sumW;



						float* pAge_tmp = (float*)(Age_tmp[m].data);
						pAge_tmp[i + j*modelWidth] = (temp_age[0][m] + temp_age[1][m] + temp_age[2][m] + temp_age[3][m]) / sumW;


#else
						m_Mean_Temp[m][idxNow] = temp_mean[3][m] / sumW;
						m_Age_Temp[m][idxNow] = temp_age[3][m] / sumW;




#endif
					}
				}
			}

			// For Variance
			{
				float temp_var[4][NUM_MODELS];
				memset(temp_var, 0, sizeof(float)* 4 * NUM_MODELS);

				float temp_var_extra[4][NUM_MODELS];
				memset(temp_var_extra, 0, sizeof(float)* 4 * NUM_MODELS);


				float temp_var_max = 0;

				float R_diff, G_diff, B_diff;

#ifdef WARP_MIX
				// Horizontal Neighbor
				if (di != 0){
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight)
					{

						int idxNew = idx_new_i + idx_new_j*modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m)
						{
							float* pVar = (float*)(Var_[m].data);
							temp_var[0][m] = w_H * (pVar[idxNew]);
							float* pMean_tmp = (float*)(Mean_tmp[m].data);
							float* pMean_ = (float*)(Mean_[m].data);

							R_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3], (int)2);
							G_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 1] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3 + 1], (int)2);
							B_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 2] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3 + 2], (int)2);


							temp_var_extra[0][m] = (R_diff + G_diff + B_diff)/3;


						}
					}
				}

				// Vertical Neighbor
				if (dj != 0){
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight)
					{

						int idxNew = idx_new_i + idx_new_j*modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m)
						{
							float* pVar = (float*)(Var_[m].data);
							temp_var[1][m] = w_V * (pVar[idxNew]);
							
							float* pMean_tmp = (float*)(Mean_tmp[m].data);
							float* pMean_ = (float*)(Mean_[m].data);

							R_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3], (int)2);
							G_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 1] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3 + 1], (int)2);
							B_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 2] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3 + 2], (int)2);


							temp_var_extra[1][m] = (R_diff + G_diff + B_diff) / 3;

						}
					}
				}


				// HV Neighbor
				if (dj != 0 && di != 0){
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight){

						int idxNew = idx_new_i + idx_new_j*modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m){


							float* pVar = (float*)(Var_[m].data);
							temp_var[2][m] = w_HV * (pVar[idxNew]);
	//						temp_var_extra[2][m] = (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2));

							float* pMean_tmp = (float*)(Mean_tmp[m].data);
							float* pMean_ = (float*)(Mean_[m].data);

							R_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3], (int)2);
							G_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 1] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3 + 1], (int)2);
							B_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 2] - pMean_[idx_new_i * 3 + idx_new_j*modelWidth * 3 + 2], (int)2);


							temp_var_extra[2][m] = (R_diff + G_diff + B_diff) / 3;





						}
					}
				}
#endif
				// Self
				if (idxNewI >= 0 && idxNewI < curModelWidth && idxNewJ >= 0 && idxNewJ < curModelHeight){

					int idxNew = idxNewI + idxNewJ*modelWidth;
					for (int m = 0; m < NUM_MODELS; ++m){

						float* pVar = (float*)(Var_[m].data);

						temp_var[3][m] = w_self * (pVar[idxNew]);
			//			temp_var_extra[3][m] = (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2));

						float* pMean_tmp = (float*)(Mean_tmp[m].data);
						float* pMean_ = (float*)(Mean_[m].data);

						R_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3] - pMean_[idxNewI * 3 + idxNewJ*modelWidth * 3], (int)2);
						G_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 1] - pMean_[idxNewI * 3 + idxNewJ*modelWidth * 3 + 1], (int)2);
						B_diff = pow(pMean_tmp[3 * i + j*modelWidth * 3 + 2] - pMean_[idxNewI * 3 + idxNewJ*modelWidth * 3 + 2], (int)2);


						temp_var_extra[3][m] = (R_diff + G_diff + B_diff) / 3;



					}
				}

				if (sumW > 0){
					for (int m = 0; m < NUM_MODELS; ++m){
#ifdef WARP_MIX

						float w_1, w_2, w_3, w_4;


						if (var_mix_method == 1)
						{
							w_1 = w_H;
							w_2 = w_V;
							w_3 = w_HV;
							w_4 = w_self;
							sumW2 = w_1 + w_2 + w_3 + w_4;
						}
						if (var_mix_method == 2)
						{
							w_1 = w_H * w_H;
							w_2 = w_V * w_V;
							w_3 = w_HV * w_HV;
							w_4 = w_self * w_self;
							sumW2 = w_1 + w_2 + w_3 + w_4;
						}
						if (var_mix_method == 3)
						{
							w_1 = 0.0;
							w_2 = 0.0;
							w_3 = 0.0;
							w_4 = 0.0;
							sumW2 = 1.0;
						}

						float* pVar_tmp = (float*)(Var_tmp[m].data);

						pVar_tmp[i + j*modelWidth] = (temp_var[0][m] + temp_var[1][m] + temp_var[2][m] + temp_var[3][m]) / sumW + (w_1 * temp_var_extra[0][m] + w_2 * temp_var_extra[1][m] + w_3 * temp_var_extra[2][m] + w_4 * temp_var_extra[3][m]) / sumW2;




#else
						m_Var_Temp[m][idxNow] = (temp_var[3][m]) / sumW;
#endif
					}
				}

			}

			// Limitations and Exceptions
			for (int m = 0; m < NUM_MODELS; ++m)
			{
				float* pVar_tmp = (float*)(Var_tmp[m].data);
				pVar_tmp[i + j*modelWidth] = MAX(pVar_tmp[i + j*modelWidth], MIN_BG_VAR);
			}
			if (idxNewI < 1 || idxNewI >= modelWidth - 1 || idxNewJ < 1 || idxNewJ >= modelHeight - 1){
				for (int m = 0; m < NUM_MODELS; ++m)
				{

					float* pVar_tmp = (float*)(Var_tmp[m].data);
					pVar_tmp[i + j*modelWidth] = INIT_BG_VAR;

					float* pAge_tmp = (float*)(Age_tmp[m].data);
					pAge_tmp[i + j*modelWidth] = 0;
				}
			}
			else {
				for (int m = 0; m < NUM_MODELS; ++m)
				{
					float* pVar_tmp = (float*)(Var_tmp[m].data);
					float* pAge_tmp = (float*)(Age_tmp[m].data);
					pAge_tmp[i + j*modelWidth] = MIN(pAge_tmp[i + j*modelWidth] * exp(-VAR_DEC_RATIO*MAX(0.0, pVar_tmp[i + j*modelWidth] - VAR_MIN_NOISE_T)), MAX_BG_AGE);
				}
			}
		}
	}

}




void CProbModel::update(Mat pOutputImg, Mat ROI, int frame_num, float s, bool bInit)
{





	unsigned char* pOut;
	if (!pOutputImg.empty())
	{
		pOutputImg = Mat::zeros(obsWidth, obsHeight, CV_8UC1);
		pOut = (unsigned char*)(pOutputImg.data);
	}


	int curModelWidth = modelWidth;
	int curModelHeight = modelHeight;


	unsigned char* pCur = (unsigned char*)(m_Cur.data);
	unsigned char* pROI = (unsigned char*)(ROI.data);

	//int margin = nMargin / BLOCK_SIZE;
	int margin = 0;

	int remainder = 0;
	int skip_level = 1;


	if (bInit == true)
	{
		skip_level = 1;
	}
	else if (s < 2)
	{
		skip_level = 2;
	}


//	int obsWidthStep = m_Cur->widthStep;


	//////////////////////////////////////////////////////////////////////////
	// 141117. Check the illumination change.

	Scalar curMM = mean(m_Cur);
	float cur_mean_R = curMM.val[0];
	float cur_mean_G = curMM.val[1];
	float cur_mean_B = curMM.val[2];

	Scalar BG_act_mean = mean(Mean_[0]);


	float mean_diff_R = 0;
	float mean_diff_G = 0;
	float mean_diff_B = 0;




#ifdef AUTO_AE
	mean_diff_R = cur_mean_R - BG_act_mean.val[0];
	mean_diff_G = cur_mean_G - BG_act_mean.val[1];
	mean_diff_B = cur_mean_B - BG_act_mean.val[2];


#endif

#ifndef LEARN_SKIP
	frame_num = remainder;
	skip_level = 1;
#endif



	// PRL17: 2의 배수일 때만, 모델 업데이트. 테스트는 매번 하고. 
	if (frame_num % skip_level == remainder)
	{
		//////////////////////////////////////////////////////////////////////////
		// Find Matching Model
		ModelIdx = Mat::zeros(modelHeight, modelWidth, CV_8UC1);

		for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++)
		{
			for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++)
			{


				// base (i,j) for this block
				int idx_base_i;
				int idx_base_j;
				idx_base_i = ((float)bIdx_i) * BLOCK_SIZE_X;
				idx_base_j = ((float)bIdx_j) * BLOCK_SIZE_Y;

				float cur_mean_R = 0;
				float cur_mean_G = 0;
				float cur_mean_B = 0;
				float elem_cnt = 0;


				for (int jj = 0; jj < BLOCK_SIZE_Y; ++jj)
				{
					for (int ii = 0; ii < BLOCK_SIZE_X; ++ii)
					{

						int idx_i = idx_base_i + ii;
						int idx_j = idx_base_j + jj;

						if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
							continue;

						cur_mean_R += pCur[idx_i * 3 + idx_j*obsWidth * 3];
						cur_mean_G += pCur[idx_i * 3 + idx_j*obsWidth * 3 + 1];
						cur_mean_B += pCur[idx_i * 3 + idx_j*obsWidth * 3 + 2];

						elem_cnt += 1.0;

					}
				}//loop for pixels
				cur_mean_R /= elem_cnt;
				cur_mean_G /= elem_cnt;
				cur_mean_B /= elem_cnt;


				//////////////////////////////////////////////////////////////////////////
				// Make Oldest Idx to 0 (swap)
				int oldIdx = 0;
				float oldAge = 0;
				for (int m = 0; m < NUM_MODELS; ++m)
				{
					float* pAge_tmp = (float*)(Age_tmp[m].data);
					float fAge = pAge_tmp[bIdx_i + bIdx_j*modelWidth];

					if (fAge >= oldAge)
					{
						oldIdx = m;
						oldAge = fAge;
					}
				}
				if (oldIdx != 0)
				{
					float* pMean_tmp_0 = (float*)(Mean_tmp[0].data);
					float* pMean_tmp_1 = (float*)(Mean_tmp[1].data);

					
					
					pMean_tmp_0[bIdx_i*3 + bIdx_j*modelWidth*3] = pMean_tmp_1[bIdx_i*3 + bIdx_j*modelWidth*3];
					pMean_tmp_0[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] = pMean_tmp_1[bIdx_i*3 + bIdx_j*modelWidth*3 + 1];
					pMean_tmp_0[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] = pMean_tmp_1[bIdx_i*3 + bIdx_j*modelWidth*3 + 2];

					pMean_tmp_1[bIdx_i*3 + bIdx_j*modelWidth*3] = cur_mean_R;
					pMean_tmp_1[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] = cur_mean_G;
					pMean_tmp_1[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] = cur_mean_B;



					float* pVar_tmp_0 = (float*)(Var_tmp[0].data);
					float* pVar_tmp_1 = (float*)(Var_tmp[1].data);

					pVar_tmp_0[bIdx_i + bIdx_j*modelWidth] = pVar_tmp_1[bIdx_i + bIdx_j*modelWidth];
					pVar_tmp_1[bIdx_i + bIdx_j*modelWidth] = INIT_BG_VAR;

					float* pAge_tmp_0 = (float*)(Age_tmp[0].data);
					float* pAge_tmp_1 = (float*)(Age_tmp[1].data);

					pAge_tmp_0[bIdx_i + bIdx_j*modelWidth] = pAge_tmp_1[bIdx_i + bIdx_j*modelWidth];
					pAge_tmp_1[bIdx_i + bIdx_j*modelWidth] = 0.0;




				}

				//////////////////////////////////////////////////////////////////////////
				// Select Model 
				// Check Match against 0
				float* pMean_tmp = (float*)(Mean_tmp[0].data);
				float* pVar_tmp = (float*)(Var_tmp[0].data);
				float R_diff, G_diff, B_diff, diff_avg;

				R_diff = pow(cur_mean_R - pMean_tmp[bIdx_i * 3 + bIdx_j*modelWidth * 3] + mean_diff_R, (int)2);
				G_diff = pow(cur_mean_G - pMean_tmp[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 1] + mean_diff_G, (int)2);
				B_diff = pow(cur_mean_B - pMean_tmp[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 2] + mean_diff_B, (int)2);
				diff_avg = (R_diff + G_diff + B_diff) / 3;

				unsigned char* pModelIdx = (unsigned char*)(ModelIdx.data);

				if (diff_avg < VAR_THRESH_MODEL_MATCH * pVar_tmp[bIdx_i + bIdx_j*modelWidth])
				{
					pModelIdx[bIdx_i + bIdx_j*modelWidth] = 0;
				}


				// Check Match against 1
				else
				{
					pMean_tmp = (float*)(Mean_tmp[1].data);
					pVar_tmp = (float*)(Var_tmp[1].data);
					//		float R_diff, G_diff, B_diff, diff_avg;

					R_diff = pow(cur_mean_R - pMean_tmp[bIdx_i * 3 + bIdx_j*modelWidth * 3] + mean_diff_R, (int)2);
					G_diff = pow(cur_mean_G - pMean_tmp[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 1] + mean_diff_G, (int)2);
					B_diff = pow(cur_mean_B - pMean_tmp[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 2] + mean_diff_B, (int)2);
					diff_avg = (R_diff + G_diff + B_diff) / 3;


					if (diff_avg < VAR_THRESH_MODEL_MATCH * pVar_tmp[bIdx_i + bIdx_j*modelWidth])
					{
						pModelIdx[bIdx_i + bIdx_j*modelWidth] = 1;
					}

						// If No match, set 1 age to zero and match = 1
					else
					{
						pModelIdx[bIdx_i + bIdx_j*modelWidth] = 1;

						float* pAge_tmp = (float*)(Age_tmp[1].data);
						pAge_tmp[bIdx_i + bIdx_j*modelWidth] = 0;
					}

				}

				

			}


		}// loop for models

	}



	// update with current observation
	float obs_mean_R[NUM_MODELS];
	float obs_mean_G[NUM_MODELS];
	float obs_mean_B[NUM_MODELS];



	float bg_mean = 0;



	for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++)
	{
		for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++)
		{
			// base (i,j) for this block
			int idx_base_i;
			int idx_base_j;
			idx_base_i = ((float)bIdx_i) * BLOCK_SIZE_X;
			idx_base_j = ((float)bIdx_j) * BLOCK_SIZE_Y;

			float elem_cnt = 0;
			float sample_mean = 0;

			unsigned char* pModelIdx = (unsigned char*)(ModelIdx.data);
			unsigned char nMatchIdx = pModelIdx[bIdx_i + bIdx_j*modelWidth];



			// obtain observation mean
			memset(obs_mean_R, 0, sizeof(float)*NUM_MODELS);
			memset(obs_mean_G, 0, sizeof(float)*NUM_MODELS);
			memset(obs_mean_B, 0, sizeof(float)*NUM_MODELS);

			int nElemCnt[NUM_MODELS];	memset(nElemCnt, 0, sizeof(int)*NUM_MODELS);


			for (int jj = 0; jj < BLOCK_SIZE_Y; ++jj)
			{
				for (int ii = 0; ii < BLOCK_SIZE_X; ++ii)
				{

					int idx_i = idx_base_i + ii;
					int idx_j = idx_base_j + jj;

					if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
						continue;


					// 141119. Sampling map consideration.
					//						if (pROI[idx_i + idx_j*obsWidthStep] > 0)
					if (true)
					{
						obs_mean_R[nMatchIdx] += pCur[idx_i * 3 + idx_j*obsWidth * 3];
						obs_mean_G[nMatchIdx] += pCur[idx_i * 3 + idx_j*obsWidth * 3 + 1];
						obs_mean_B[nMatchIdx] += pCur[idx_i * 3 + idx_j*obsWidth * 3 + 2];
						++nElemCnt[nMatchIdx];
					}

				}
			}



			for (int m = 0; m < NUM_MODELS; ++m)
			{


				if (nElemCnt[m] <= 0)
				{
					float* pMean = (float*)(Mean_[m].data);
					float* pMean_tmp = (float*)(Mean_tmp[m].data);

					pMean[bIdx_i*3 + bIdx_j*modelWidth*3] = MAX(MIN(pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3] + mean_diff_R, 255), 0);
					pMean[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] = MAX(MIN(pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] + mean_diff_G, 255), 0);
					pMean[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] = MAX(MIN(pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] + mean_diff_B, 255), 0);


				}
				else
				{
					// learning rate for this block
					float* pAge_tmp = (float*)(Age_tmp[m].data);

					float age = pAge_tmp[bIdx_i + bIdx_j*modelWidth];
					float alpha = age / (age + 1.0);

					obs_mean_R[m] /= ((float)nElemCnt[m]);
					obs_mean_G[m] /= ((float)nElemCnt[m]);
					obs_mean_B[m] /= ((float)nElemCnt[m]);
					// update with this mean

					if (frame_num % skip_level == remainder)
					{
						// 160830. add
						if (obs_mean_R[m] == 0)
							alpha = 1.0;


						float* pMean = (float*)(Mean_[m].data);
						float* pMean_tmp = (float*)(Mean_tmp[m].data);

						pMean[bIdx_i*3 + bIdx_j*modelWidth*3] = MAX(MIN(alpha * (pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3] + mean_diff_R) + (1.0 - alpha) * obs_mean_R[m], 255), 0);
						pMean[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] = MAX(MIN(alpha * (pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] + mean_diff_G) + (1.0 - alpha) * obs_mean_G[m], 255), 0);
						pMean[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] = MAX(MIN(alpha * (pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] + mean_diff_B) + (1.0 - alpha) * obs_mean_B[m], 255), 0);





					}
					else if (age < 1.0)
					{
						float* pMean = (float*)(Mean_[m].data);
						
						pMean[bIdx_i*3 + bIdx_j*modelWidth*3] = MAX(MIN(obs_mean_R[m], 255), 0);
						pMean[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] = MAX(MIN(obs_mean_G[m], 255), 0);
						pMean[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] = MAX(MIN(obs_mean_B[m], 255), 0);




					}


					else
					{
						float* pMean = (float*)(Mean_[m].data);
						float* pMean_tmp = (float*)(Mean_tmp[m].data);
						
						pMean[bIdx_i * 3 + bIdx_j*modelWidth * 3] = MAX(MIN(pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3] + mean_diff_R, 255), 0);
						pMean[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 1] = MAX(MIN(pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3 + 1] + mean_diff_G, 255), 0);
						pMean[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 2] = MAX(MIN(pMean_tmp[bIdx_i*3 + bIdx_j*modelWidth*3 + 2] + mean_diff_B, 255), 0);




					}
				}
			}

			//float* pMean_tmp_0 = (float*)(Mean_tmp[0].data);

			//bg_mean_R += pMean_tmp_0[bIdx_i * 3 + bIdx_j*modelWidth * 3];
			//bg_mean_G += pMean_tmp_0[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 1];
			//bg_mean_B += pMean_tmp_0[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 2];
		}
	}
	//m_bgMean = bg_mean / (double)(curModelHeight * curModelWidth);





	float obs_var[NUM_MODELS];

	for (int bIdx_j = 0; bIdx_j < curModelHeight; bIdx_j++)
	{
		for (int bIdx_i = 0; bIdx_i < curModelWidth; bIdx_i++)
		{


			// TODO: OPTIMIZE THIS PART SO THAT WE DO NOT CALCULATE THIS (LUT)
			// base (i,j) for this block
			int idx_base_i;
			int idx_base_j;
			idx_base_i = ((float)bIdx_i) * BLOCK_SIZE_X;
			idx_base_j = ((float)bIdx_j) * BLOCK_SIZE_Y;


			unsigned char* pModelIdx = (unsigned char*)(ModelIdx.data);
			unsigned char nMatchIdx = pModelIdx[bIdx_i + bIdx_j*modelWidth];




			int nSampleblock = 0;
			int nFG = 0;

			// obtain observation variance
			memset(obs_var, 0, sizeof(float)*NUM_MODELS);
			int nElemCnt[NUM_MODELS];
			memset(nElemCnt, 0, sizeof(int)*NUM_MODELS);
			for (int jj = 0; jj < BLOCK_SIZE_Y; ++jj)
			{
				for (int ii = 0; ii < BLOCK_SIZE_X; ++ii)
				{



					int idx_i = idx_base_i + ii;
					int idx_j = idx_base_j + jj;
					nElemCnt[nMatchIdx]++;

					if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
					{
						continue;
					}


					float* pMean = (float*)(Mean_[0].data); // * for test, we use only acting model!!!
					float* pAge_tmp_0 = (float*)(Age_tmp[0].data);
					float* pVar_0 = (float*)(Var_[0].data);
					float* pVar_tmp_0 = (float*)(Var_tmp[0].data);
					float R_diff, G_diff, B_diff, diff_avg;
					
					//		float R_diff, G_diff, B_diff, diff_avg;

					R_diff = pow(pCur[idx_i*3 + idx_j*obsWidth*3] - pMean[bIdx_i * 3 + bIdx_j*modelWidth * 3], (int)2);
					G_diff = pow(pCur[idx_i*3 + idx_j*obsWidth*3 + 1] - pMean[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 1], (int)2);
					B_diff = pow(pCur[idx_i*3 + idx_j*obsWidth*3 + 2] - pMean[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 2], (int)2);
					diff_avg = (R_diff + G_diff + B_diff) / 3;
					float pixelDist = 0.0;
					pixelDist += diff_avg;

					// 180803. For max-variance setting
					float* pMean_sel = (float*)(Mean_[nMatchIdx].data);
					float R_diff2, G_diff2, B_diff2, diff_avg2;

					R_diff2 = pow(pCur[idx_i * 3 + idx_j*obsWidth * 3] - pMean_sel[bIdx_i * 3 + bIdx_j*modelWidth * 3], (int)2);
					G_diff2 = pow(pCur[idx_i * 3 + idx_j*obsWidth * 3 + 1] - pMean_sel[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 1], (int)2);
					B_diff2 = pow(pCur[idx_i * 3 + idx_j*obsWidth * 3 + 2] - pMean_sel[bIdx_i * 3 + bIdx_j*modelWidth * 3 + 2], (int)2);
					diff_avg2 = (R_diff2 + G_diff2 + B_diff2) / 3;
					float pixelDist2 = 0.0;
					pixelDist2 += diff_avg2;





	//				m_DistImg[idx_i + idx_j*obsWidthStep] = pow(pCur[idx_i + idx_j*obsWidthStep] - m_Mean[0][bIdx_i + bIdx_j*modelWidth], (int)2);

					if (!pOutputImg.empty() && pAge_tmp_0[bIdx_i + bIdx_j*modelWidth] > 1)
					{

						//unsigned char valOut = diff_avg > VAR_THRESH_FG_DETERMINE * pVar_0[bIdx_i + bIdx_j*modelWidth] ? 255 : 0;
						unsigned char valOut = diff_avg > VAR_THRESH_FG_DETERMINE * pVar_tmp_0[bIdx_i + bIdx_j*modelWidth] ? 255 : 0;

						if (true)
							pOut[idx_i + idx_j*obsWidth] = valOut;
					}

					obs_var[nMatchIdx] = MAX(obs_var[nMatchIdx], pixelDist2); 
				}
			}

			for (int m = 0; m < NUM_MODELS; ++m)
			{
				
				float* pAge = (float*)(Age_[m].data);
				float* pVar = (float*)(Var_[m].data);
				float* pVar_tmp = (float*)(Var_tmp[m].data);
				float* pAge_tmp = (float*)(Age_tmp[m].data);

				if (nElemCnt[m] > 0)
				{
					
					
					float age = pAge_tmp[bIdx_i + bIdx_j*modelWidth];
					float alpha = age / (age + 1.0);

					// update with this variance
					if (age == 0)
					{				
						pVar[bIdx_i + bIdx_j*modelWidth] = MAX(obs_var[m], INIT_BG_VAR);

					}

					else
					{
						
						float alpha_var = alpha;//MIN(alpha, 1.0 - MIN_NEW_VAR_OBS_PORTION);
						pVar[bIdx_i + bIdx_j*modelWidth] = alpha_var * pVar_tmp[bIdx_i + bIdx_j*modelWidth] + (1.0 - alpha_var) * obs_var[m];
						pVar[bIdx_i + bIdx_j*modelWidth] = MAX(pVar[bIdx_i + bIdx_j*modelWidth], MIN_BG_VAR);
					}

					// Update Age
					pAge[bIdx_i + bIdx_j*modelWidth] = pAge_tmp[bIdx_i + bIdx_j*modelWidth] + 1.0;
					pAge[bIdx_i + bIdx_j*modelWidth] = MIN(pAge[bIdx_i + bIdx_j*modelWidth], MAX_BG_AGE);
				}

				else
				{
					pVar[bIdx_i + bIdx_j*modelWidth] = pVar_tmp[bIdx_i + bIdx_j*modelWidth];
					pAge[bIdx_i + bIdx_j*modelWidth] = pAge_tmp[bIdx_i + bIdx_j*modelWidth];
				}

			}


		}
	}


	// 170623. Tmp img copy to original
	for (int m = 0; m < 2; m++)
	{
		Mean_[m].copyTo(Mean_tmp[m]);
		Var_[m].copyTo(Var_tmp[m]);
		Age_[m].copyTo(Age_tmp[m]);
	}
	




}


void CProbModel::update_vibe(Mat pOutputImg, Mat AGE, Mat FGS, int nFrame, float fZero_ratio)
{
	// if fZero_ratio is negative, motion compensation module is off...



	unsigned char* pOut;
	if (!pOutputImg.empty())
	{
	//	cvSet(pOutputImg, CV_RGB(128, 128, 128)); // modify the Vibe initial values... 

//		pOutputImg = Mat::zeros(obsWidth, obsHeight, CV_8UC1);
		pOut = (unsigned char*)(pOutputImg.data);
	}




	// 170425. Auto exposure control
	// 141117. Check the illumination change.
	Mat cur_img;
	m_Cur.copyTo(cur_img);
	cur_img.convertTo(cur_img, CV_32F);


	Scalar curMM = mean(cur_img);
	float cur_mean = curMM.val[0];

	float mean_diff = 0;
	int skip_level = 1;

	/*
	if (s < 2)
	{
	skip_level = 2;
	}*/


	unsigned char* pCur = (unsigned char*)(m_Cur.data);
	float* pAge = (float*)(AGE.data); // Age map
	//float* pFGS = (float*)FGS->imageData; // Foreground speed after homography 

	Mat dist_map = Mat::zeros(obsHeight, obsWidth, CV_32F);
	float* pDist = (float*)dist_map.data;



	int obsWidthStep = m_Cur.cols;
	int roi_idx;


//	Mat mat_Cur = cvarrToMat(m_Cur);


	// ViBE parameter : default for fixed camera. 
	int N = 20;
	float R = 20.0;
	int num_min = 2;
	int pi_max = 16;
	int pi = pi_max;
	unsigned char background = 0;
	unsigned char foreground = 255;
	int randNum = 0;
	int randNum2 = 0;
	int randNum3 = 0;
	int n_nbr = 2;
	int x_nbr = 0;
	int y_nbr = 0;


	// Test parameter for moving camera.
//	int N = 20;
	/*int R = 20;
	int num_min = 2;
	int pi_max = 16;
	int pi = pi_max;
	unsigned char background = 0;
	unsigned char foreground = 255;
	int randNum = 0;
	int randNum2 = 0;
	int randNum3 = 0;
	int n_nbr = 4;
	int x_nbr = 0;
	int y_nbr = 0;*/
	//		int s_space = 0; // for moving camera window... 

	int i, j;
	int num_min2 = 2;


	//if (fZero_ratio < 0)
	//{
	//	if (nFrame < 20)
	//		mat_Cur.copyTo(m_Samples[nFrame]);
	//}
	//else
	//{
	//	 170424. For moving camera
	//	 first frae
	//	if (nFrame < 1)
	//	{
	//		for (int m = 0; m < 20; m++)
	//		{
	//			mat_Cur.copyTo(m_Samples[m]);
	//		}
	//	}
	//}
	

	if (nFrame < 1)
	{
		for (int m = 0; m < 20; m++)
		{
			m_Cur.copyTo(m_Samples[m]);
		}
	
	}


	if (fZero_ratio < 0 && nFrame < 20) // when motion compensation is off.
		m_Cur.copyTo(m_Samples[nFrame]);


	else
	{

		// BG display code... 
		Mat BGtmp;
		Mat BGMean(m_Samples[0].rows, m_Samples[0].cols, CV_32FC3);

		BGMean.setTo(Scalar(0, 0, 0, 0));
		for (int mm = 0; mm < 20; mm++)
		{
			// Convert the input images to CV_64FC3 ...
			m_Samples[mm].convertTo(BGtmp, CV_32FC3);
			BGMean += BGtmp;
		}

		// Convert back to CV_8UC3 type, applying the division to get the actual mean
		BGMean.convertTo(BGMean, CV_32F, 1. / 20);

		//Mat BGMean_disp;
		//BGMean.convertTo(BGMean_disp, CV_8U);
		//imshow("BG_10", m_Samples[10]); // display 10-th bg image
		//imshow("BG_mean", BGMean_disp); // display bg mean


		Mat BGVar(m_Samples[0].rows, m_Samples[0].cols, CV_32FC3);
		BGVar.setTo(Scalar(0, 0, 0, 0));
		// 170524. Variance across the channel.
		for (int m = 0; m < 20; m++)
		{
			m_Samples[m].convertTo(BGtmp, CV_32FC3);
			BGVar += abs(BGMean - BGtmp);
		}

		//BGVar.convertTo(BGVar, CV_32F, 1. / 20);

		//Mat BGVar_disp;
		//BGVar.convertTo(BGVar_disp, CV_8U);
		//imshow("BG_Var", BGVar_disp); // display bg mean





		Scalar BG_avg;
		Scalar Cur_avg;

		BG_avg = mean(BGMean);
		Cur_avg = mean(cur_img);
		Scalar color_diff = Cur_avg - BG_avg;
		//			Scalar color_diff = cv::Scalar(0, 0, 0, 0);
		int s_space = 0;

		//			if (nZeros == 0)
		//				s_space = 0; // if camera is not moved, the neighber search is off.


		for (j = 0; j < obsHeight; j++)
		{
			for (i = 0; i < obsWidth; i++)
			{
				int count = 0;

				// newly appearing region updage.
				if (pAge[i + j*obsWidth] <= 2) // if age == 2, update with current frame
				{
					for (int m = 0; m < 20; m++)
					{
						unsigned char* samples = (unsigned char*)(m_Samples[m].data);
						samples[3 * i + j*obsWidth * 3] = pCur[3 * i + j*obsWidth * 3];
						samples[3 * i + 1 + j*obsWidth * 3] = pCur[3 * i + 1 + j*obsWidth * 3];
						samples[3 * i + 2 + j*obsWidth * 3] = pCur[3 * i + 2 + j*obsWidth * 3];
					}

					pOut[i + j*obsWidth] = background;
				}
				// other region - FG/BG test.
				else
				{
					for (int m = 0; m < 20; m++)
					{
						unsigned char* samples = (unsigned char*)(m_Samples[m].data);
						samples[3 * i + j*obsWidth * 3] = MAX(MIN(samples[3 * i + j*obsWidth * 3] + color_diff.val[0], 255), 0);
						samples[3 * i + 1 + j*obsWidth * 3] = MAX(MIN(samples[3 * i + 1 + j*obsWidth * 3] + color_diff.val[1], 255), 0);
						samples[3 * i + 2 + j*obsWidth * 3] = MAX(MIN(samples[3 * i + 2 + j*obsWidth * 3] + color_diff.val[2], 255), 0);
					}


					float distance = 0;
					float R_dist, G_dist, B_dist, R_dist_tmp, G_dist_tmp, B_dist_tmp;

					//float fg_speed = pFGS[i + j*obsWidth];
					//						s_space = MIN(round(1 + 7*exp(-fg_speed / 1.25)), 12);
					
					
					if (fZero_ratio > 0)
						s_space = 1; // 170518. at least need 1 if homograhy estimation is used... 


					float* pBGVar = (float*)(BGVar.data);
					float Var_R, Var_G, Var_B;

					// PRL 17. Reprojection error for adaptive search space. 
					for (int index = 0; index < N; index++)
					{
						for (int y_nbr_r = -s_space; y_nbr_r <= s_space; y_nbr_r++)
						{
							for (int x_nbr_r = -s_space; x_nbr_r <= s_space; x_nbr_r++)
							{
								x_nbr = x_nbr_r + i;
								y_nbr = y_nbr_r + j;

								if (x_nbr >= 0 && x_nbr < obsWidth && y_nbr >= 0 && y_nbr < obsHeight)
								{

									unsigned char* samples = (unsigned char*)(m_Samples[index].data);
									R_dist = (float)(samples[3 * x_nbr + y_nbr*obsWidth * 3]) - (float)(pCur[3 * i + j*obsWidth * 3]);
									G_dist = (float)(samples[3 * x_nbr + 1 + y_nbr*obsWidth * 3]) - (float)(pCur[3 * i + 1 + j*obsWidth * 3]);
									B_dist = (float)(samples[3 * x_nbr + 2 + y_nbr*obsWidth * 3]) - (float)(pCur[3 * i + 2 + j*obsWidth * 3]);

									Var_R = (float)(pBGVar[3 * x_nbr + y_nbr*obsWidth * 3]);
									Var_G = (float)(pBGVar[3 * x_nbr + y_nbr*obsWidth * 3 + 1]);
									Var_B = (float)(pBGVar[3 * x_nbr + y_nbr*obsWidth * 3 + 2]);


									// 170518. Distance with Color bias...
									/*				R_dist = (float)(samples[3 * x_nbr + y_nbr*obsWidth * 3]) - (float)(pCur[3 * i + j*obsWidth * 3]) - color_diff.val[0];
									G_dist = (float)(samples[3 * x_nbr + 1 + y_nbr*obsWidth * 3]) - (float)(pCur[3 * i + 1 + j*obsWidth * 3]) - color_diff.val[1];
									B_dist = (float)(samples[3 * x_nbr + 2 + y_nbr*obsWidth * 3]) - (float)(pCur[3 * i + 2 + j*obsWidth * 3]) - color_diff.val[2];*/


									// 170428. Fail... Too much slow... just mean matcing is enough... 
									/*			R_dist = pow((float)(samples[3 * x_nbr + y_nbr*obsWidth * 3]), gamma_b) * alpha_b - (float)(pCur[3 * i + j*obsWidth * 3]);
									G_dist = pow((float)(samples[3 * x_nbr + 1 + y_nbr*obsWidth * 3]), gamma_g) * alpha_g - (float)(pCur[3 * i + 1 + j*obsWidth * 3]);
									B_dist = pow((float)(samples[3 * x_nbr + 2 + y_nbr*obsWidth * 3]), gamma_r) * alpha_r - (float)(pCur[3 * i + 2 + j*obsWidth * 3]);

									*/
									//distance = R_dist*R_dist + G_dist*G_dist + B_dist*B_dist;

									// 170524. if variance is considered, the edge region overcompressed...
					//				distance = R_dist*R_dist / Var_R + G_dist*G_dist / Var_G + B_dist*B_dist / Var_B;



									// 171010. Distance measure is modified...
									distance = abs(R_dist) + abs(G_dist) + abs(B_dist);



									//		R = 40 * (1.5 - (1 / (1 + exp(-fg_speed))));

									// 170524. exponential is exetremly slow... just several case paramter is used.
									/*			if (fg_speed < 1.0)
									R = 60;
									else if (fg_speed < 3.0)
									R = 40;
									else
									R = 30;*/




									float dist_th = 3.0;

									if (distance < R * dist_th)
										count++;

									if (count >= num_min)
										pOut[i + j*obsWidth] = background;
									break;
								}
							}
						}
					}


					// Update Equation 
					// Background 


					if (count >= num_min)
					{
						//			pOut[i + j*obsWidth] = background;




						pi = MIN(round(pAge[i + j*obsWidth]), pi_max);
						randNum = rand() % (pi - 1);

						if (randNum == 0)
						{
							randNum = rand() % (N - 1);
							unsigned char* samples = (unsigned char*)(m_Samples[randNum].data);
							samples[3 * i + j*obsWidth * 3] = pCur[3 * i + j*obsWidth * 3];
							samples[3 * i + 1 + j*obsWidth * 3] = pCur[3 * i + 1 + j*obsWidth * 3];
							samples[3 * i + 2 + j*obsWidth * 3] = pCur[3 * i + 2 + j*obsWidth * 3];

							//							m_Samples[index].at<Vec3b>(j, i) = mat_Cur.at<Vec3b>(j, i);


						}


						//					pi = MIN(round(pAge[i + j*obsWidth]), pi_max);
						// 170524. 2x update speed for neighbor update... but is it effective?? 
						pi = MIN(round(pAge[i + j*obsWidth]) / 2 + 1, pi_max / 2);
						randNum = rand() % (pi - 1);
						if (randNum == 0)
						{
							x_nbr = -1;
							while (x_nbr < 0 || x_nbr >= obsWidth)
							{
								randNum2 = rand() % (n_nbr)-n_nbr / 2;
								x_nbr = i + randNum2;

							}

							y_nbr = -1;
							while (y_nbr < 0 || y_nbr >= obsHeight)
							{
								randNum2 = rand() % (n_nbr)-n_nbr / 2;
								y_nbr = j + randNum2;

							}

							randNum3 = rand() % (N - 1);
							unsigned char* samples = (unsigned char*)(m_Samples[randNum3].data);

							samples[3 * x_nbr + y_nbr*obsWidth * 3] = pCur[3 * i + j*obsWidth * 3];
							samples[3 * x_nbr + 1 + y_nbr*obsWidth * 3] = pCur[3 * i + 1 + j*obsWidth * 3];
							samples[3 * x_nbr + 2 + y_nbr*obsWidth * 3] = pCur[3 * i + 2 + j*obsWidth * 3];

						}
					}
					/*	else if (count >= num_min)
					{
					pOut[i + j*obsWidth] = 128;
					}*/


					else
						pOut[i + j*obsWidth] = foreground;
				}

			}
		}



	}





}


//void CProbModel::run(Mat curFrame, bool bMoving)
//{
//
//	
//	ipl_imgRe = cvCloneImage(&(IplImage)curFrame);
//	double s = sum(matCurFrame_re).val[0]; // for skipping the black frames
//
//	if (bInit_vibe == false && s > 0)
//	{
//		m_ProbModel.init(ipl_imgRe, 1);
//		m_Age_map = Mat::ones(matCurFrame_re.rows, matCurFrame_re.cols, CV_32FC1) * 16;
//		ipl_Age = cvCloneImage(&(IplImage)m_Age_map);
//		fZero_ratio = -1.0;
//		bInit_vibe = true;
//
//	}
//	else if (bInit_vibe == true)
//	{
//		m_ProbModel.m_Cur = ipl_imgRe;
//		m_ProbModel.update_vibe(ipl_foreground, ipl_Age, ipl_FGS, fIdx, fZero_ratio);
//		m_fg_map = cvarrToMat(ipl_foreground);
//		// 170926. Gaussian Smoothing for Clear background...
//		//	GaussianBlur(m_fg_map, m_fg_map, Size(3, 3), 0, 0);
//		//		medianBlur(m_fg_map, m_fg_map, 3);
//
//		imshow("foreground", m_fg_map);
//
//
//	}
//
//
//
//
//
//
//}