#include "ActionClassifier.h"
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace jm
{

CActionClassifier::CActionClassifier()
	: bInit_(false)
{
}


CActionClassifier::~CActionClassifier()
{
	Finalize();
}

void CActionClassifier::Initialize(stParamAction &_stParam/*, std::string _strModelPath*/)
{
	if (bInit_) { Finalize(); }

	stActionParam_ = _stParam;
	bInit_ = true;

	//****************************************************
	//    Visualization related
	//****************************************************
	bVisualizeResult_ = stActionParam_.bVisualize;
	strVisWindowName_ = "Detection result";

	//****************************************************
	//    SVM Load related
	//****************************************************
	

	/* Select train_xml file */
	std::string _strModelPath;
	if (stActionParam_.bNormalize)
	{
		if(stActionParam_.bUsingDisparity) { _strModelPath = "../model\\norm_using_disparity.xml"; }

		else{ _strModelPath = "../model\\normalize_model.xml"; }
	}

	else 
	{
		if(stActionParam_.bUsingDisparity) { _strModelPath = "../model\\not_norm_using_disparity.xml"; }
		
		else { _strModelPath = "../model\\not_normalize_model.xml"; }
	}

	/* Load SVM */
	svm = cv::ml::SVM::create();
	svm = cv::ml::SVM::load<cv::ml::SVM>(_strModelPath);

}

void CActionClassifier::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }

	svm->clear();
}

void CActionClassifier::UpdatePoseletUsingTrack()
{
	//active Poselet update
	std::deque<CPoselet*> newActivePoselets;
	std::deque<CPoselet*> newPendingPoselets;

	//---------------------------------------------------
	// MATCHING STEP 01: active Poselet <-> keypoints
	//---------------------------------------------------
	for (std::vector<hj::CObjectInfo>::iterator objectIter = curTrackResult_.objectInfos.begin();
		objectIter != curTrackResult_.objectInfos.end();)
	{
		bool match = false;
		for (int poseletIdx = 0; poseletIdx < activePoselets_.size(); poseletIdx++)
		{
			CPoselet *curPoselet = activePoselets_[poseletIdx];

			if (curPoselet->id != objectIter->id) { continue; }

			curPoselet->nEndFrame = this->nCurrentFrameIdx_;         // TODO: EndFrame과 last update은 달라야 한다 interpolation 해야하는 지점들 찾기위함(나중에 둘중 하나는 없앨지도..)
			curPoselet->duration++;
			curPoselet->vectorObjInfo.push_back(*objectIter);
			newActivePoselets.push_back(curPoselet);

			objectIter = curTrackResult_.objectInfos.erase(objectIter);
			match = true;
			break;
		}

		if (match) { continue; }
		objectIter++;
	}

	// Update pending poselet
	for (std::deque<CPoselet*>::iterator poseIter = activePoselets_.begin();
		poseIter != activePoselets_.end(); poseIter++)
	{
		if ((*poseIter)->nEndFrame == this->nCurrentFrameIdx_) { continue; }
		newPendingPoselets.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 02: pending Poselet <-> keypoints
	//---------------------------------------------------
	for (std::vector<hj::CObjectInfo>::iterator objectIter = curTrackResult_.objectInfos.begin();
		objectIter != curTrackResult_.objectInfos.end();)
	{
		bool match = false;
		for (int poseletIdx = 0; poseletIdx < pendingPoselets_.size(); poseletIdx++)
		{
			CPoselet *curPoselet = pendingPoselets_[poseletIdx];

			if (curPoselet->id != objectIter->id) { continue; }

			////// inactive에서 active로 업데이트 될때 update된 프레임 -1이 이전에 최근 업뎃과 같지 않으면 interpolation하는 작업
			//////interpolation ( TODO: 방법 수정)
			//int pendingTime = this->nCurrentFrameIdx_ - curPoselet->nEndFrame;
			//for (int interpolFrame = 1; interpolFrame < pendingTime; interpolFrame++)
			//{
			//	curPoselet->duration++;

			//	int nKeypoint = 0;
			//	hj::stKeyPoint tmpPoint;
			//	std::vector<hj::stKeyPoint> tmpPoints;
			//	hj::CObjectInfo tmpObjInfo;

			//	for (std::vector<hj::stKeyPoint>::iterator pointIter = objectIter->keyPoint.begin();
			//		pointIter != objectIter->keyPoint.end(); pointIter++, nKeypoint++)
			//	{
			//		tmpPoint.x = 
			//			//( (pendingTime - interpolFrame) * curPoselet->vectorPose.back()[nKeypoint].x + interpolFrame * pointIter->x) / pendingTime;
			//			((pendingTime - interpolFrame) * curPoselet->vectorObjInfo.back().keyPoint.at(nKeypoint).x + interpolFrame * pointIter->x) / pendingTime;

			//		tmpPoint.y = 
			//			//( (pendingTime - interpolFrame) * curPoselet->vectorPose.back()[nKeypoint].y + interpolFrame * pointIter->y) / pendingTime;
			//			((pendingTime - interpolFrame) * curPoselet->vectorObjInfo.back().keyPoint.at(nKeypoint).y + interpolFrame * pointIter->y) / pendingTime;
			//		tmpPoint.confidence = 0;
			//		tmpPoints.push_back(tmpPoint);
			//	}

			//	tmpObjInfo.id = objectIter->id;
			//	tmpObjInfo.keyPoint = tmpPoints;

			//	curPoselet->vectorObjInfo.push_back(tmpObjInfo);
			//	//curPoselet->vectorPose.push_back(tmpPoints);
			//}


			curPoselet->nEndFrame  = this->nCurrentFrameIdx_;         
			curPoselet->duration++;
			curPoselet->vectorObjInfo.push_back(*objectIter);
			newActivePoselets.push_back(curPoselet);


			objectIter = curTrackResult_.objectInfos.erase(objectIter);
			match = true;
			break;
		}

		if (match) { continue; }
		objectIter++;
	}

	// Update pending poselet
	for (std::deque<CPoselet*>::iterator poseIter = pendingPoselets_.begin();
		poseIter != pendingPoselets_.end(); poseIter++)
	{
		if ((*poseIter)->nEndFrame == this->nCurrentFrameIdx_) { continue; }
		newPendingPoselets.push_back((*poseIter));
	}


	//---------------------------------------------------
	// MATCHING STEP 03: Generation Poselet
	//---------------------------------------------------
	for (int idx = 0; idx < curTrackResult_.objectInfos.size(); idx++)
	{
		CPoselet newPoselet;

		newPoselet.id = curTrackResult_.objectInfos[idx].id;
		newPoselet.nStartFrame = this->nCurrentFrameIdx_;
		newPoselet.nEndFrame = this->nCurrentFrameIdx_;
		newPoselet.duration = 1;
		newPoselet.vectorObjInfo.push_back(curTrackResult_.objectInfos[idx]);

		this->listCPoselets_.push_back(newPoselet);
		newActivePoselets.push_back(&this->listCPoselets_.back());
	}

	//------------------------------------------------
	// MATCHING STEP 04: POSELET TERMINATION
	//------------------------------------------------
	for (std::deque<CPoselet*>::iterator poseIter = newPendingPoselets.begin(); poseIter != newPendingPoselets.end();)
	{
		if ((*poseIter)->nEndFrame + stActionParam_.nMaxPendingFrame < nCurrentFrameIdx_)
		{
			poseIter = newPendingPoselets.erase(poseIter);
			continue;
		}
		poseIter++;
	}

	activePoselets_ = newActivePoselets;
	pendingPoselets_ = newPendingPoselets;

}


// Load python SVM train model 
void CActionClassifier::Detect(std::deque<CPoselet*> _activePoselets, hj::CTrackResult *_curTrackResult)
{	
	listActionResult.clear();

	for (std::deque<CPoselet*>::iterator poseletIter = _activePoselets.begin();
		poseletIter != _activePoselets.end(); poseletIter++)
	{
		stActionResult curActionResult;
		curActionResult.trackId = (*poseletIter)->id;

		//30frame 채워지지 않았다면 이전 Frame의 결과를 받아오기
		if ((*poseletIter)->vectorObjInfo.size() < stActionParam_.nPoseLength)
		{
			for (std::vector<stActionResult>::iterator prevResultIter = actionResult_.actionResults.begin();
				prevResultIter != actionResult_.actionResults.end(); prevResultIter++)
			{
				if (prevResultIter->trackId != (*poseletIter)->id) { continue; }

				curActionResult.bActionDetect = prevResultIter->bActionDetect;
			}
		}

		//30frame 만족한다면 Detect해서 결과 저장.
		else 
		{
			cv::Mat sampleMat, tmpMat;
			std::vector<hj::stKeyPoint> preKeypoint;

			for (CAction::iterator objIter = (*poseletIter)->vectorObjInfo.begin();
				objIter != (*poseletIter)->vectorObjInfo.end(); objIter++)
			{
				double normDist = CalcNormDist(objIter->keyPoint);

				if (normDist == 1) 
				{ 
					printf("here"); 
				}

				// save keypoint vector
				for (std::vector<hj::stKeyPoint>::iterator pointIter = objIter->keyPoint.begin();
					pointIter != objIter->keyPoint.end(); pointIter++)
				{
					sampleMat.push_back( (pointIter->x - objIter->keyPoint.at(1).x) / normDist);
					sampleMat.push_back( (pointIter->y - objIter->keyPoint.at(1).y) / normDist);
				}

				// disparity 사용안하면 저장하지 않기
				if (!stActionParam_.bUsingDisparity) { continue; }

				// save disparity vector
				std::vector<hj::stKeyPoint> curKeypoint = objIter->keyPoint;
				for (int keyPtsSize = 0; keyPtsSize < objIter->keyPoint.size(); keyPtsSize++)
				{
					// 30 frame중 가장 첫 frame이라면 0으로 모두 저장
					if (objIter == (*poseletIter)->vectorObjInfo.begin()) 
					{ 
						sampleMat.push_back(double(0));  // x 변위
						sampleMat.push_back(double(0));  // y 변위 
						
					}

					else
					{
						sampleMat.push_back(curKeypoint.at(keyPtsSize).x - preKeypoint.at(keyPtsSize).x);
						sampleMat.push_back(curKeypoint.at(keyPtsSize).y - preKeypoint.at(keyPtsSize).y);
					}

				}
				preKeypoint = curKeypoint;
				
			}

			//// disparity 사용안하면 저장하지 않기
			//if (stActionParam_.bUsingDisparity)
			//{
			//	for (CAction::iterator objIter = (*poseletIter)->vectorObjInfo.begin();
			//		objIter != (*poseletIter)->vectorObjInfo.end(); objIter++)
			//	{
			//		// save disparity vector
			//		std::vector<hj::stKeyPoint> curKeypoint = objIter->keyPoint;
			//		for (int keyPtsSize = 0; keyPtsSize < objIter->keyPoint.size(); keyPtsSize++)
			//		{
			//			// 30 frame중 가장 첫 frame이라면 0으로 모두 저장
			//			if (objIter == (*poseletIter)->vectorObjInfo.begin())
			//			{
			//				sampleMat.push_back(double(0));  // x 변위
			//				sampleMat.push_back(double(0));  // y 변위 

			//			}

			//			else
			//			{
			//				sampleMat.push_back(curKeypoint.at(keyPtsSize).x - preKeypoint.at(keyPtsSize).x);
			//				sampleMat.push_back(curKeypoint.at(keyPtsSize).y - preKeypoint.at(keyPtsSize).y);
			//			}

			//		}
			//		preKeypoint = curKeypoint;
			//	}
			//}

			sampleMat.convertTo(tmpMat, CV_32FC1);
			int res = svm->predict(tmpMat.t());
			printf("frame: %d  trackID: %d response: %d\n", nCurrentFrameIdx_, (*poseletIter)->id, res);

			if (res)
				curActionResult.bActionDetect = true;
			else
				curActionResult.bActionDetect = false;
		}			


		listActionResult.push_back(curActionResult);

		
	}

	
}


// Train in opencv SVM model 
void CActionClassifier::TrainSVM(std::string _saveModelPath)
{
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

	//TODO: Train process
	//demand Labeling 

	svm->save(_saveModelPath);

}

void CActionClassifier::EliminationStepSize()
{
	for (std::deque<CPoselet*>::iterator poseletIter = activePoselets_.begin();
		poseletIter != activePoselets_.end(); poseletIter++)
	{
		if ((*poseletIter)->vectorObjInfo.size() < stActionParam_.nPoseLength) { continue; }

		//elimination front step size object info.
		for (int poseIdx = 0; poseIdx < stActionParam_.nStepSize; poseIdx++)
		{
			(*poseletIter)->vectorObjInfo.pop_front();
			(*poseletIter)->nStartFrame++;
		}
		(*poseletIter)->bActionDetect = false;
	}

}


// when input action vector 
CActionResultSet CActionClassifier::Run(hj::CTrackResult *_curTrackResult, cv::Mat _curFrame, int _frameIdx)
{
	nCurrentFrameIdx_ = _frameIdx;
	matDetectResult_ = _curFrame.clone();
	curTrackResult_ = *_curTrackResult;
	UpdatePoseletUsingTrack();
	

	Detect(activePoselets_, _curTrackResult);
	ResultPackaging();
	
	

	/* visualize */
	if (bVisualizeResult_) { Visualize(_curTrackResult); }

	EliminationStepSize();

	return this->actionResult_;
}


// Normalize relate
double CActionClassifier::CalcNormDist(std::vector<hj::stKeyPoint> _curKeyoints)
{
	if (!stActionParam_.bNormalize) { return 1; }

	double dist, leftDist, rightDist;
	hj::stKeyPoint neckPoint     = _curKeyoints.at(1);
	hj::stKeyPoint rightShoulder = _curKeyoints.at(2);
	hj::stKeyPoint leftShoulder  = _curKeyoints.at(5);
	
	leftDist = std::pow((neckPoint.x - leftShoulder.x), 2) + std::pow((neckPoint.y - leftShoulder.y), 2);
	rightDist = std::pow((neckPoint.x - rightShoulder.x), 2) + std::pow((neckPoint.y - rightShoulder.y), 2);

	dist = (leftDist > rightDist) ? leftDist : rightDist;

	return std::sqrt(dist);
}


void CActionClassifier::ResultPackaging()
{
	time_t timePackaging = clock();
	actionResult_.frameIdx = nCurrentFrameIdx_;
	actionResult_.timeStamp = (unsigned int)timePackaging;
	actionResult_.actionResults.clear();

	actionResult_.actionResults = listActionResult;

}

// Result visualization
void CActionClassifier::Visualize(hj::CTrackResult *_curTrackResult)
{
	/* frame information */
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "%04d", this->nCurrentFrameIdx_);
	cv::rectangle(matDetectResult_, cv::Rect(5, 2, 100, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(matDetectResult_, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));

	for (std::vector<stActionResult>::iterator resultIter = listActionResult.begin();
		resultIter != listActionResult.end(); resultIter++)
	{
		if (!resultIter->bActionDetect) { continue; }

		for (int index = 0; index < _curTrackResult->objectInfos.size(); index++)
		{
			if (_curTrackResult->objectInfos.at(index).id != resultIter->trackId) { continue; }

			cv::rectangle(
				matDetectResult_,
				_curTrackResult->objectInfos.at(index).box,
				cv::Scalar(0, 0, 255), 1);

			char strDetectResult[100];
			sprintf_s(strDetectResult, "Throwing Detected(%d person)", resultIter->trackId);
			cv::putText(matDetectResult_, strDetectResult, cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

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

	cv::imshow(strVisWindowName_, matDetectResult_);
	cv::waitKey(1);
	matDetectResult_.release();
}



}