#include "ResultIntegration.h"
#include <opencv2/imgproc/imgproc.hpp>



namespace jm
{

CResultIntegration::CResultIntegration()
	:bInit_(false)
{
}


CResultIntegration::~CResultIntegration()
{
	Finalize();
}

void CResultIntegration::Initialize(stParamResult &_stParam)
{
	if (bInit_) { Finalize(); }

	stResultParam_ = _stParam;
	bInit_ = true;


	// visualization related
	bVisualizeResult_ = stResultParam_.bVisualize;
	strVisWindowName_ = "Final result";
	if (bVisualizeResult_)
	{
		vecColors_ = GenerateColors(400);
	}

}

void CResultIntegration::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }
}

CDetectResultSet CResultIntegration::Run(hj::CTrackResult *_trackResult, 
	CThrownResultSet *_throwResult,
	jm::CActionResultSet *_actionResult, 
	cv::Mat _curFrame, 
	int _frameIdx)
{
	nCurrentFrameIdx_ = _frameIdx;
	matResult_ = _curFrame.clone();

	/* save result to member variable */
	curTrackResult_ = *_trackResult;
	curActionResult_ = *_actionResult;
	curThrowResult_ = *_throwResult;

	Integrate(/*_trackResult, _actionResult*/);

	if (bVisualizeResult_) { Visualize(); }

	return this->integratedResult_;
}


void CResultIntegration::Integrate(/*hj::CTrackResult *_trackResult, jm::CActionResultSet *_actionResult*/)
{
	assert(curTrackResult_.objectInfos.size() == curActionResult_.actionResults.size());
	//assert(curTrackResult_.objectInfos.size() == curThrowResult_.throwResults.size());   //넣지 말아야 하는건가?!
	
	integratedResult_.detectResults.clear();                     // 이부분 다르게 수정할 방법은?
	integratedResult_.frameIdx = this->nCurrentFrameIdx_;

	
	for (std::vector<hj::CObjectInfo>::iterator objIter = curTrackResult_.objectInfos.begin();
		objIter != curTrackResult_.objectInfos.end(); objIter++)
	{

		//stDetectResult *curDetectResult;   //이거 왜 에러나지(?) 생각해보기
		stDetectResult curDetectResult;

		curDetectResult.trackId = objIter->id;
		curDetectResult.keyPoint = objIter->keyPoint;
		curDetectResult.box = objIter->box;
		curDetectResult.headBox = objIter->headBox;
		
		// 두개를 각각 저장하는 방법 외에 한번에 저장하는 방식은 없을까? 

		for (std::vector<CThrowDetector>::iterator throwIter = curThrowResult_.throwResults.begin();
			throwIter != curThrowResult_.throwResults.end(); throwIter++)
		{
			if (objIter->id != throwIter->trackId) { continue; }

			curDetectResult.bKCFDetect  = throwIter->m_bThw_warning;
			curDetectResult.bMASKDetect = throwIter->m_bThw_warning2;
		}

		for (std::vector<stActionResult>::iterator actionIter = curActionResult_.actionResults.begin();
			actionIter != curActionResult_.actionResults.end(); actionIter++)
		{
			if (objIter->id != actionIter->trackId) { continue; }

			curDetectResult.bActionDetect = actionIter->bActionDetect;
		}


		integratedResult_.detectResults.push_back(curDetectResult);
	}
	

	
}

void CResultIntegration::Visualize()
{
	/* frame information */
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "%04d", this->nCurrentFrameIdx_);
	cv::rectangle(matResult_, cv::Rect(5, 2, 100, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(matResult_, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));

	int num_detect = 0;

	/* Result */
	for (std::vector<stDetectResult>::iterator resultIter = integratedResult_.detectResults.begin();
		resultIter != integratedResult_.detectResults.end(); resultIter++)
	{

		DrawBoxWithID(
			matResult_, 
			resultIter->box, 
			resultIter->trackId, 
			1,
			0, 
			getColorByID(resultIter->trackId, &vecColors_));

		if (resultIter->bActionDetect)
		{

			cv::rectangle(
				matResult_, 
				cv::Rect((int)resultIter->box.x+ 7, (int)resultIter->box.y - 12, 14, 14), 
				cv::Scalar(0, 0, 255), 
				CV_FILLED);

			num_detect++;
			char strDetectResult[100];
			sprintf_s(strDetectResult, "SVM Detected(%d person)", resultIter->trackId);
			cv::putText(matResult_, strDetectResult, cv::Point(10, 100 * num_detect), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

		}

		if (resultIter->bKCFDetect)
		{

			cv::rectangle(
				matResult_, 
				cv::Rect((int)resultIter->box.x+ 21, (int)resultIter->box.y - 12, 14, 14), 
				cv::Scalar(0, 255, 0), 
				CV_FILLED);

			num_detect++;
			char strDetectResult[100];
			sprintf_s(strDetectResult, "KCF Detected(%d person)", resultIter->trackId);
			cv::putText(matResult_, strDetectResult, cv::Point(10, 100 * num_detect), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

		}

		if (resultIter->bMASKDetect)
		{

			cv::rectangle(
				matResult_, 
				cv::Rect((int)resultIter->box.x+ 35, (int)resultIter->box.y - 12, 14, 14),
				cv::Scalar(255, 0, 0), 
				CV_FILLED);

			num_detect++;
			char strDetectResult[100];
			sprintf_s(strDetectResult, "Mask Detected(%d person)", resultIter->trackId);
			cv::putText(matResult_, strDetectResult, cv::Point(10, 100 * num_detect), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

		}

	}




	////---------------------------------------------------
	//// RECORD
	////---------------------------------------------------
	//if (bVideoWriterInit_)
	//{
	//	IplImage *currentFrame = new IplImage(matTrackingResult_);
	//	cvWriteFrame(videoWriter_, currentFrame);
	//	delete currentFrame;
	//}

	cv::namedWindow(strVisWindowName_);

	cv::imshow(strVisWindowName_, matResult_);
	cv::waitKey(1);
	matResult_.release();
}

// 이 함수를 어떻게 없애지... Track에서 받아오는게 좋을듯 한데..(재정의)
unsigned int CResultIntegration::DrawBoxWithID(
	cv::Mat &imageFrame,
	cv::Rect box,
	unsigned int nID,
	int lineStyle,
	int fontSize,
	cv::Scalar curColor)
{
	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while (tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}
	if (0 == fontSize)
	{
		cv::rectangle(imageFrame, box, curColor, 1 + lineStyle);
		cv::rectangle(imageFrame, cv::Rect((int)box.x, (int)box.y - 13, 7 * labelLength, 14), curColor, CV_FILLED);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y-4), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));

	}
	else
	{
		cv::rectangle(imageFrame, box, curColor, 1 + lineStyle);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, curColor);
	}

	return labelLength;
}


cv::Scalar CResultIntegration::getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors)
{
	if (NULL == vecColors) { return cv::Scalar(255, 255, 255); }
	unsigned int colorIdx = nID % vecColors->size();
	return (*vecColors)[colorIdx];
}

std::vector<cv::Scalar> CResultIntegration::GenerateColors(unsigned int numColor)
{
	double golden_ratio_conjugate = 0.618033988749895;
	//double hVal = (double)std::rand()/(INT_MAX);
	double hVal = 0.0;
	std::vector<cv::Scalar> resultColors;
	resultColors.reserve(numColor);
	for (unsigned int colorIdx = 0; colorIdx < numColor; colorIdx++)
	{
		hVal += golden_ratio_conjugate;
		hVal = std::fmod(hVal, 1.0);
		resultColors.push_back(hsv2rgb(hVal, 0.5, 0.95));
	}
	return resultColors;
}


cv::Scalar CResultIntegration::hsv2rgb(double h, double s, double v)
{
	int h_i = (int)(h * 6);
	double f = h * 6 - (double)h_i;
	double p = v * (1 - s);
	double q = v * (1 - f * s);
	double t = v * (1 - (1 - f) * s);
	double r, g, b;
	switch (h_i)
	{
	case 0: r = v; g = t; b = p; break;
	case 1: r = q; g = v; b = p; break;
	case 2: r = p; g = v; b = t; break;
	case 3: r = p; g = q; b = v; break;
	case 4: r = t; g = p; b = v; break;
	case 5: r = v; g = p; b = q; break;
	default:
		break;
	}

	return cv::Scalar((int)(r * 255), (int)(g * 255), (int)(b * 255));
}

}