#include <limits>
#include <assert.h>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "HungarianMethod.h"
#include "MTTracker.h"


namespace hj
{

CMTTracker::CMTTracker()
	: bInit_(false)
	, bVisualizeResult_(false)
	, strVisWindowName_("")
{
}


CMTTracker::~CMTTracker()
{
	Finalize();
}


/************************************************************************
 Method Name: Initialize
 Description:
	- Initialize the multiple target tracker with the given parameters.
 Input Arguments:
	- _stParams: structure of parameters
 Return Values:
	- None
************************************************************************/
void CMTTracker::Initialize(stParamTrack &_stParams)
{
	// check duplicated initialization
	if (bInit_) { Finalize(); }

	stParam_ = _stParams;
	stParam_.dImageRescaleRecover = 1.0 / stParam_.dImageRescale;

	nCurrentFrameIdx_ = 0;	
	
	trackingResult_.frameIdx = nCurrentFrameIdx_;
	trackingResult_.objectInfos.clear();

	nInputWidth_  = _stParams.nImageWidth;
	nInputHeight_ = _stParams.nImageHeight;

	// detection related
	vecKeypoints_.clear();

	// tracker related
	nNewTrackletID_ = 0;
	listCTracklet_.clear();
	queueActiveTracklets_.clear();

	// trajectory related
	nNewTrajectoryID_ = 0;

	// input related
	sizeBufferImage_ = cv::Size(
		(int)((double)nInputWidth_  * stParam_.dImageRescale),
		(int)((double)nInputHeight_ * stParam_.dImageRescale));
	matGrayImage_ = cv::Mat(nInputHeight_, nInputWidth_, CV_8UC1);
	cImageBuffer_.set(stParam_.nBackTrackingLength);	

	// feature tracking related		
	featureDetector_          = cv::AgastFeatureDetector::create();
	matFeatureExtractionMask_ = cv::Mat(sizeBufferImage_, CV_8UC1, cv::Scalar(0));

	// visualization related
	bVisualizeResult_ = stParam_.bVisualize;
	strVisWindowName_ = "Tracking result";
	if (bVisualizeResult_)
	{
		vecColors_ = GenerateColors(400);
	}

	// record
	bRecord_ = stParam_.bVideoRecord;
	bVideoWriterInit_ = false;
	if (bRecord_) {
		strRecordPath_ = stParam_.strVideoRecordPath;

		// get time
		time_t curTimer = time(NULL);
		struct tm timeStruct;
		localtime_s(&timeStruct, &curTimer);

		// make file name
		char resultOutputFileName[256];
		sprintf_s(resultOutputFileName, "%s_%02d%02d%02d_%02d%02d%02d.avi",
			strRecordPath_.c_str(),
			timeStruct.tm_year + 1900,
			timeStruct.tm_mon + 1,
			timeStruct.tm_mday,
			timeStruct.tm_hour,
			timeStruct.tm_min,
			timeStruct.tm_sec);

		// init video writer
		CvSize imgSize;
		imgSize.width = stParam_.nImageWidth;
		imgSize.height = stParam_.nImageHeight;
		videoWriter_ = cvCreateVideoWriter(resultOutputFileName, CV_FOURCC('M', 'J', 'P', 'G'), 30, imgSize, 1);
		bVideoWriterInit_ = true;
	}

	// initialization flag
	bInit_ = true;
}


/************************************************************************
 Method Name: Finalize
 Description:
	- Terminate the class with memory clean up.
 Input Arguments:
	- None
 Return Values:
	- None
************************************************************************/
void CMTTracker::Finalize(void)
{	
	if (!bInit_) { return; }

	/* detection related */
	this->vecKeypoints_.clear();

	/* tracker related */	
	listCTracklet_.clear();
	queueActiveTracklets_.clear();

	/* input related */
	cImageBuffer_.clear();
	if (!matGrayImage_.empty()) { matGrayImage_.release(); }
	if (!matTrackingResult_.empty()) { matTrackingResult_.release(); }

	/* matching related */
	arrKeyPointToTrackletMatchingCost_.clear();
	arrInterTrackletMatchingCost_.clear();

	/* result related */
	trackingResult_.objectInfos.clear();

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }

	// record
	if (bRecord_)
	{
		cvReleaseVideoWriter(&this->videoWriter_);
	}

	/* initialization flag */
	bInit_ = false;
}


/************************************************************************
 Method Name: Track
 Description:
	- Run the tracking algorithm on the current input frame and detections.
 Input Arguments:
	- _vecInputDetections: Input detections of the current frame
	- _curFrame: Current input frame image
	- _frameIdx: Current frame index
 Return Values:
	- CTrackResult: Tracking result of the current frame
************************************************************************/
CTrackResult CMTTracker::Track(
	KeyPointsSet _vecCurKeyPoints,
	cv::Mat _curFrame, 
	int _frameIdx)
{
	time_t timeStartTrack = clock();

	// validate input size
	assert(bInit_ && _curFrame.rows == matGrayImage_.rows
		&& _curFrame.cols == matGrayImage_.cols);
	
	//matHeadPatch_ = _curFrame.clone();
	this->EstimateHeads(_vecCurKeyPoints);
	vecKeypoints_ = _vecCurKeyPoints;

	// frame buffering
	nCurrentFrameIdx_ = _frameIdx;
	matGrayImage_ = _curFrame;
	cImageBuffer_.insert_resize(matGrayImage_, sizeBufferImage_);
	if (!matTrackingResult_.empty()) { matTrackingResult_.release(); }
	matTrackingResult_ = _curFrame.clone();
	//cv::cvtColor(_curFrame, matTrackingResult_, cv::COLOR_GRAY2BGR);
	
	UpdateTrajectories(vecKeypoints_, activeTrajectories_, inactiveTrajectories_);

	/* result packaging */
	ResultPackaging();	
	trackingResult_.procTime = clock() - timeStartTrack;

	/* visualize */
	if (bVisualizeResult_) { VisualizeResult(); }

	return this->trackingResult_;
}


/************************************************************************
 Method Name: MatchingKeypointsAndTracklets
 Description:
	- Do matching between detections and tracklets. Then, update tracklets.
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
TrackletPtQueue CMTTracker::UpdateTracklets(
	TrackletPtQueue _keyPointsTracklets,
	TrackletPtQueue _activeTracklets)
{
	/////////////////////////////////////////////////////////////////////////////
	// CALCULATE MATCHING COSTS
	/////////////////////////////////////////////////////////////////////////////
	std::vector<float> arrKeyPointToTrackletMatchingCost_(
		_keyPointsTracklets.size() * _activeTracklets.size(),
		std::numeric_limits<float>::infinity());
	
	// to determine occlusion
	std::vector<std::deque<int>> trackletIdxOfFeaturePoints(_keyPointsTracklets.size());

	//---------------------------------------------------
	// COST WITH BI-DIRECTIONAL TRACKING
	//---------------------------------------------------
	for (int t = 0; t < _activeTracklets.size(); t++)
	{
		CTracklet *curTracklet = _activeTracklets[t];
		for (int k = 0, costPos = t; 
			k < _keyPointsTracklets.size();
			k++, costPos += (int)_activeTracklets.size())
		{
			// validate with backward tracking result
			if (!hj::CheckOverlap(curTracklet->currentBox(), _keyPointsTracklets[k]->currentBox()))
				continue;

			// check the candidate owning tracklets of keypoints
			for (int p = 0; p < curTracklet->featurePointsHistory.back().size(); p++)
			{
				if (_keyPointsTracklets[k]->currentBox().contains(curTracklet->featurePointsHistory.back()[p]))
				{
					trackletIdxOfFeaturePoints[k].push_back(t);
					break;
				}
			}		

			// determine the possible longest comparison interval
			size_t lengthForCompare = std::min((size_t)stParam_.nBackTrackingLength, 
				std::min(curTracklet->queueKeyPoints.size(), _keyPointsTracklets[k]->queueKeyPoints.size()));
			
			// croll tracker boxes for comparison (reverse ordering)
			size_t numBoxCopy = lengthForCompare - 1;
			std::vector<cv::Rect2d> vecTrackerBoxes;
			vecTrackerBoxes.reserve(lengthForCompare);			
			for (int i = 0; i < std::min(lengthForCompare, curTracklet->queueKeyPoints.size()); i++)
			{
				vecTrackerBoxes.push_back(curTracklet->queueKeyPoints[curTracklet->queueKeyPoints.size() - i - 1].bbox);
			}			

			// box cost with bidirectional tracking
			double boxCost = 0.0;
			cv::Rect2d keypointBox, trackletBox;
			int keypointBoxPos = 0;			
			for (int boxIdx = 0; boxIdx < lengthForCompare; boxIdx++)
			{
				keypointBoxPos = (int)_keyPointsTracklets[k]->queueKeyPoints.size() - 1 - boxIdx;
				keypointBox = _keyPointsTracklets[k]->queueKeyPoints[keypointBoxPos].bbox;
				trackletBox = vecTrackerBoxes[boxIdx];

				if (!hj::CheckOverlap(keypointBox, trackletBox) // rejection criterion
					|| stParam_.dMaxBoxDistance < BoxCenterDistanceWRTScale(keypointBox, trackletBox)
					|| stParam_.dMinBoxOverlapRatio > hj::OverlappedArea(keypointBox, trackletBox) / std::min(keypointBox.area(), trackletBox.area())
					|| stParam_.dMaxBoxCenterDiffRatio * std::max(keypointBox.width, trackletBox.width) < hj::NormL2(hj::Center(keypointBox) - hj::Center(trackletBox)))
				{
					boxCost = std::numeric_limits<double>::infinity();
					break;
				}
				boxCost += BoxCenterDistanceWRTScale(trackletBox, keypointBox);
			}
			if (std::numeric_limits<double>::infinity() == boxCost) { continue; }
			boxCost /= (double)lengthForCompare;

			arrKeyPointToTrackletMatchingCost_[costPos] = (float)boxCost;
		}
	}

	//---------------------------------------------------
	// OCCLUSION HANDLING
	//---------------------------------------------------	
	// If a detection box contains feature points from more than one tracker, we examine whether there exists a
	// dominant tracker or not. If there is no dominant tracker, that means the ownership of the detection is 
	// not clear, we set the scores between that detection and trackers to infinite. This yields termination of
	// all related trackers.

	int numFeatureFromMajorTracker = 0,
		numFeatureFromCurrentTracker = 0,
		majorTrackerIdx = 0,
		currentTrackerIdx = 0;

	for (size_t detectIdx = 0, costPos = 0; 
		detectIdx < _keyPointsTracklets.size();
		detectIdx++, costPos += _activeTracklets.size())
	{
		if (0 == trackletIdxOfFeaturePoints[detectIdx].size()) { continue; }

		// find dominant tracker of the detection
		numFeatureFromMajorTracker = numFeatureFromCurrentTracker = 0;
		majorTrackerIdx = trackletIdxOfFeaturePoints[detectIdx].front();
		currentTrackerIdx = trackletIdxOfFeaturePoints[detectIdx].front();
		for (int featureIdx = 0; featureIdx < trackletIdxOfFeaturePoints[detectIdx].size(); featureIdx++)
		{
			if (currentTrackerIdx == trackletIdxOfFeaturePoints[detectIdx][featureIdx])
			{
				// we assume that the same tracker indices in 'trackletIdxOfFeaturePoints' are grouped together
				numFeatureFromCurrentTracker++;
				continue;
			}

			if (numFeatureFromCurrentTracker > numFeatureFromMajorTracker)
			{
				majorTrackerIdx = currentTrackerIdx;
				numFeatureFromMajorTracker = numFeatureFromCurrentTracker;
			}
			currentTrackerIdx = trackletIdxOfFeaturePoints[detectIdx][featureIdx];
			numFeatureFromCurrentTracker = 0;
		}

		// case 1: only one tracker has its features points in the detecion box
		if (trackletIdxOfFeaturePoints[detectIdx].front() == currentTrackerIdx)
		{
			continue;
		}

		// case 2: there is a domninant tracker among related trackers
		if (numFeatureFromMajorTracker > trackletIdxOfFeaturePoints[detectIdx].size() * stParam_.dMinOpticalFlowMajorityRatio)
		{
			continue;
		}

		// case 3: more than one trackers are related and there is no dominant tracker
		for (size_t infCostPos = costPos; infCostPos < costPos + _activeTracklets.size(); infCostPos++)
		{
			arrKeyPointToTrackletMatchingCost_[infCostPos] = std::numeric_limits<float>::infinity();
		}
	}

	//---------------------------------------------------
	// INFINITE HANDLING
	//---------------------------------------------------
	// To ensure a proper operation of our Hungarian implementation, we convert infinite to the finite value
	// that is little bit (=100.0f) greater than the maximum finite cost in the original cost function.
	float maxCost = -1000.0f;
	for (int costIdx = 0; costIdx < arrKeyPointToTrackletMatchingCost_.size(); costIdx++)
	{
		if (!_finitef(arrKeyPointToTrackletMatchingCost_[costIdx])) { continue; }
		if (maxCost < arrKeyPointToTrackletMatchingCost_[costIdx]) { maxCost = arrKeyPointToTrackletMatchingCost_[costIdx]; }
	}
	maxCost = maxCost + 100.0f;
	for (int costIdx = 0; costIdx < arrKeyPointToTrackletMatchingCost_.size(); costIdx++)
	{
		if (_finitef(arrKeyPointToTrackletMatchingCost_[costIdx])) { continue; }
		arrKeyPointToTrackletMatchingCost_[costIdx] = maxCost;
	}


	/////////////////////////////////////////////////////////////////////////////
	// MATCHING
	/////////////////////////////////////////////////////////////////////////////
	trackingResult_.objectInfos.clear();
	CHungarianMethod cHungarianMatcher;
	cHungarianMatcher.Initialize(
		arrKeyPointToTrackletMatchingCost_,
		(unsigned int)_keyPointsTracklets.size(), 
		(unsigned int)this->queueActiveTracklets_.size());
	stMatchInfo *curMatchInfo = cHungarianMatcher.Match();
	std::vector<bool> vecKeypointMatchedWithTracklet(_keyPointsTracklets.size(), false);
	for (size_t matchIdx = 0; matchIdx < curMatchInfo->rows.size(); matchIdx++) 
	{
		if (maxCost == curMatchInfo->matchCosts[matchIdx]) { continue; }
		CTracklet *curKeypoint = _keyPointsTracklets[curMatchInfo->rows[matchIdx]];
		CTracklet *curTracklet = _activeTracklets[curMatchInfo->cols[matchIdx]];

		//---------------------------------------------------
		// MATCHING VALIDATION
		//---------------------------------------------------
		if (curTracklet->length() >= stParam_.nMaxTrackletLength) { continue; }	

		//---------------------------------------------------
		// TRACKER UPDATE
		//---------------------------------------------------		
		curTracklet->timeEnd = curKeypoint->timeStart; // keypoint tracklet has a backward direction
		curTracklet->confidence = ((curTracklet->length() - 1) * curTracklet->confidence + curKeypoint->queueKeyPoints.front().confidence) / (double)curTracklet->length();
		curTracklet->replaceKeyPoints(curKeypoint->queueKeyPoints.front(), curKeypoint->timeStart);

		vecKeypointMatchedWithTracklet[curMatchInfo->rows[matchIdx]] = true;//[matchIdx] = true; //JM

		// update features with detection (after result packaging)
		curTracklet->featurePointsHistory.back() = curKeypoint->featurePointsHistory.front();
	}
	cHungarianMatcher.Finalize();


	/////////////////////////////////////////////////////////////////////////////
	// TRACKLET GENERATION
	/////////////////////////////////////////////////////////////////////////////	
	for (int k = 0; k < _keyPointsTracklets.size(); k++)
	{
		if (vecKeypointMatchedWithTracklet[k]) { continue; }

		CTracklet newTracklet;
		newTracklet.id = this->nNewTrackletID_++;
		newTracklet.timeStart = this->nCurrentFrameIdx_;
		newTracklet.timeEnd = this->nCurrentFrameIdx_;
		newTracklet.direction = FORWARD;
		newTracklet.queueKeyPoints.push_back(_keyPointsTracklets[k]->queueKeyPoints.front());
		newTracklet.featurePointsHistory.push_back(_keyPointsTracklets[k]->featurePointsHistory.front());
		newTracklet.confidence = _keyPointsTracklets[k]->confidence;

		// generate tracklet instance
		this->listCTracklet_.push_back(newTracklet);
	}


	/////////////////////////////////////////////////////////////////////////////
	// TRACKER TERMINATION
	/////////////////////////////////////////////////////////////////////////////
	TrackletPtQueue newActiveTracklets;
	for (std::list<CTracklet>::iterator trackerIter = listCTracklet_.begin();
		trackerIter != listCTracklet_.end();
		/*trackerIter++*/)
	{
		if ((*trackerIter).timeEnd + stParam_.nMaxPendingTime < (int)nCurrentFrameIdx_)
		{
			// termination			
			trackerIter = this->listCTracklet_.erase(trackerIter);
			continue;
		}
		if ((*trackerIter).timeEnd == (int)nCurrentFrameIdx_ && (*trackerIter).length() < stParam_.nMaxTrackletLength) //JM
			newActiveTracklets.push_back(&(*trackerIter));
		trackerIter++;
	}
	queueActiveTracklets_ = newActiveTracklets;

	return queueActiveTracklets_;
}


/************************************************************************
 Method Name: TrackletToTrajectoryMatching
 Description:
	- Do matching between tracklets and trajectories. Then, update trajectories.
 Input Arguments:
	- _queueActiveTracklets: Input tracklets.
 Return Values:
	- None
************************************************************************/
void CMTTracker::TrackletToTrajectoryMatching(const TrackletPtQueue &_queueActiveTracklets)
{
	// trajectory update
	TrackletPtQueue newTracklets;
	for (size_t i = 0; i < _queueActiveTracklets.size(); i++)
	{
		if (NULL == _queueActiveTracklets[i]->ptTrajectory) 
		{
			newTracklets.push_back(_queueActiveTracklets[i]);
			continue;
		}
		CTrajectory *curTrajectory = _queueActiveTracklets[i]->ptTrajectory;
		curTrajectory->timeEnd = this->nCurrentFrameIdx_;
		curTrajectory->timeLastUpdate = curTrajectory->timeEnd;
		curTrajectory->duration = curTrajectory->timeEnd - curTrajectory->timeStart + 1;
		curTrajectory->boxes.push_back(_queueActiveTracklets[i]->currentBox());		
	}

	// delete expired trajectories and find pending trajectories
	queueActiveTrajectories_.clear();
	std::vector<CTrajectory*> vecPendedTrajectories;
	vecPendedTrajectories.reserve(listCTrajectories_.size());
	for (std::list<CTrajectory>::iterator trajIter = listCTrajectories_.begin();
		trajIter != listCTrajectories_.end();
		/*trajIter++*/)
	{
		if (trajIter->timeLastUpdate + stParam_.nMaxPendingTime < (int)this->nCurrentFrameIdx_)
		{
			// termination			
			trajIter = this->listCTrajectories_.erase(trajIter);
			continue;
		}
		if (trajIter->timeLastUpdate == this->nCurrentFrameIdx_)
			queueActiveTrajectories_.push_back(&(*trajIter));
		else
			vecPendedTrajectories.push_back(&(*trajIter));

		trajIter++;
	}

	// trajectory-to-tracklet matching
	arrInterTrackletMatchingCost_.clear();
	arrInterTrackletMatchingCost_.resize(
		vecPendedTrajectories.size() * newTracklets.size(),
		std::numeric_limits<float>::infinity());
	for (size_t trajIdx = 0; trajIdx < vecPendedTrajectories.size(); trajIdx++)
	{
		for (size_t newIdx = 0, costPos = trajIdx;
			newIdx < newTracklets.size();
			newIdx++, costPos += vecPendedTrajectories.size())
		{
			double curCost = 0.0;

			// TODO: translation + depth distance
			double distTranslate = hj::NormL2(hj::Center(vecPendedTrajectories[trajIdx]->boxes.back()) - hj::Center(newTracklets[newIdx]->currentBox()));
			if (distTranslate > stParam_.dMaxTranslationDistance)
				continue;
			curCost += distTranslate;		

			arrInterTrackletMatchingCost_[costPos] = (float)curCost;
		}
	}

	// handling infinite in the cost array
	float maxCost = -1000.0f;
	for (int costIdx = 0; costIdx < arrInterTrackletMatchingCost_.size(); costIdx++)
	{
		if (!_finitef(arrInterTrackletMatchingCost_[costIdx]))
			continue;
		if (maxCost < arrInterTrackletMatchingCost_[costIdx])
			maxCost = arrInterTrackletMatchingCost_[costIdx];
	}
	maxCost = maxCost + 100.0f;
	for (int costIdx = 0; costIdx < arrInterTrackletMatchingCost_.size(); costIdx++)
	{
		if (_finitef(arrInterTrackletMatchingCost_[costIdx]))
			continue;
		arrInterTrackletMatchingCost_[costIdx] = maxCost;
	}

	//---------------------------------------------------
	// MATCHING
	//---------------------------------------------------
	CHungarianMethod cHungarianMatcher;
	cHungarianMatcher.Initialize(arrInterTrackletMatchingCost_, (unsigned int)newTracklets.size(), (unsigned int)vecPendedTrajectories.size());
	stMatchInfo* curMatchInfo = cHungarianMatcher.Match();
	for (size_t matchIdx = 0; matchIdx < curMatchInfo->rows.size(); matchIdx++)
	{
		if (maxCost == curMatchInfo->matchCosts[matchIdx]) { continue; }
		CTracklet *curTracklet = newTracklets[curMatchInfo->rows[matchIdx]];
		CTrajectory *curTrajectory = vecPendedTrajectories[curMatchInfo->cols[matchIdx]];

		curTracklet->ptTrajectory = curTrajectory;

		// updata matched trajectory
		curTrajectory->timeEnd = this->nCurrentFrameIdx_;
		curTrajectory->timeLastUpdate = curTrajectory->timeEnd;
		curTrajectory->duration = curTrajectory->timeEnd - curTrajectory->timeStart + 1;
		curTrajectory->boxes.push_back(curTracklet->currentBox());		
		curTrajectory->tracklets.push_back(curTracklet);

		queueActiveTrajectories_.push_back(curTrajectory);
	}
	cHungarianMatcher.Finalize();

	/////////////////////////////////////////////////////////////////////////////
	// TRAJECTORY GENERATION
	/////////////////////////////////////////////////////////////////////////////	
	for (TrackletPtQueue::iterator trackletIter = newTracklets.begin();
		trackletIter != newTracklets.end();
		trackletIter++)
	{
		if ((*trackletIter)->ptTrajectory != NULL) { continue; }

		CTrajectory newTrajectory;
		newTrajectory.id = this->nNewTrajectoryID_++;
		newTrajectory.timeStart = this->nCurrentFrameIdx_;
		newTrajectory.timeEnd = this->nCurrentFrameIdx_;
		newTrajectory.timeLastUpdate = this->nCurrentFrameIdx_;
		newTrajectory.duration = 1;
		newTrajectory.boxes.push_back((*trackletIter)->currentBox());		
		newTrajectory.tracklets.push_back(*trackletIter);

		// generate trajectory instance
		this->listCTrajectories_.push_back(newTrajectory);		
		(*trackletIter)->ptTrajectory = &this->listCTrajectories_.back();
		queueActiveTrajectories_.push_back(&this->listCTrajectories_.back());
	}
}


/************************************************************************
 Method Name: ResultPackaging
 Description:
	- Packaging the tracking result into 'trackingResult_'
 Input Arguments:
	- None
 Return Values:
	- None
************************************************************************/
void CMTTracker::ResultPackaging()
{
	time_t timePackaging = clock();	
	trackingResult_.frameIdx  = nCurrentFrameIdx_;
	trackingResult_.timeStamp = (unsigned int)timePackaging;
	trackingResult_.objectInfos.clear();
	for (size_t tIdx = 0; tIdx < activeTrajectories_.size(); tIdx++)
	{
		trackingResult_.objectInfos.push_back(GetObjectInfo(activeTrajectories_[tIdx]));
	}
/*
	for (size_t tIdx = 0; tIdx < queueActiveTrajectories_.size(); tIdx++)
	{
		trackingResult_.objectInfos.push_back(GetObjectInfo(queueActiveTrajectories_[tIdx]));
	}
*/
	int cost_pos = 0;
	if (!this->trackingResult_.matMatchingCost.empty()) { trackingResult_.matMatchingCost.release(); }
	trackingResult_.matMatchingCost = 
		cv::Mat((int)vecKeypoints_.size(), (int)trackingResult_.vecTrackerRects.size(), CV_32F);
	for (int detectionIdx = 0; detectionIdx < vecKeypoints_.size(); detectionIdx++)
	{
		for (int trackIdx = 0; trackIdx < trackingResult_.vecTrackerRects.size(); trackIdx++)
		{
			this->trackingResult_.matMatchingCost.at<float>(detectionIdx, trackIdx) = arrKeyPointToTrackletMatchingCost_[cost_pos];
			cost_pos++;
		}
	}
}


/************************************************************************
 Method Name: FeatureExtraction
 Description:
	- Tracks the input feature points of the input frame in the target frame.
 Input Arguments:
	- _inputBox         : bounding box of the target on the input frame
	- _inputImage       : image containing the input feature points
	- vecFeaturePoints : target image of feature point tracking
 Return Values:
	- bool: (true) success / (false) fail
************************************************************************/
bool CMTTracker::FeatureExtraction(
	const cv::Rect2d _inputBox,
	const cv::Mat _inputImage,
	std::vector<cv::Point2f> &_vecFeaturePoints)
{
	_vecFeaturePoints.clear();

	//---------------------------------------------------
	// EXTRACT FEATURE POINTS
	//---------------------------------------------------
	std::vector<cv::KeyPoint> newKeypoints;
	cv::Rect2d rectROI = hj::CropWithSize(_inputBox, cv::Size(_inputImage.cols, _inputImage.rows));
	matFeatureExtractionMask_(rectROI) = cv::Scalar(255); // masking with the bounding box
	featureDetector_->detect(_inputImage, newKeypoints, matFeatureExtractionMask_);
	matFeatureExtractionMask_(rectROI) = cv::Scalar(0);   // restore the mask image

	if (stParam_.nMinNumFeatures > newKeypoints.size())
	{
		// it is impossible to track the target because there are insufficient number of feature points
		return false;
	}

	//---------------------------------------------------
	// EXTRACT SELECTION
	//---------------------------------------------------	
	std::random_shuffle(newKeypoints.begin(), newKeypoints.end());
	for (int pointIdx = 0; pointIdx < std::min((int)newKeypoints.size(), stParam_.nMaxNumFeatures); pointIdx++)
	{
		_vecFeaturePoints.push_back(newKeypoints[pointIdx].pt);
	}

	return true;
}


/************************************************************************
 Method Name: FeatureTracking
 Description:
	- Tracks the input feature points of the input frame in the target frame.
 Input Arguments:
	- _inputBox         : bounding box of the target at the input frame
	- _inputImage       : image containing the input feature points.
	- _targetImage      : target image of feature point tracking.
	- _vecInputFeatures : input feature points.
	- _vecOutputFeatures: points of tracking result. Actually it is an output.
	- _vecFeatureInlierIndex: index of features that are inliers of estimated motion.
	- _trackingResult   : (output) estimated box at the target frame.
 Return Values:
	- bool: (true) success / (false) fail
************************************************************************/
bool CMTTracker::FeatureTracking(
	const cv::Rect2d _inputBox,
	const cv::Mat _inputImage,
	const cv::Mat _targetImage,
	std::vector<cv::Point2f> &_vecInputFeatures,
	std::vector<cv::Point2f> &_vecOutputFeatures,
	std::vector<int>         &_vecFeatureInlierIndex,
	cv::Rect2d &_trackingResult)
{
	if (0 == _vecInputFeatures.size()) { return false; }

	_trackingResult = cv::Rect2d(0.0, 0.0, 0.0, 0.0);

	//---------------------------------------------------
	// CONVERT TO GRAY SCALE IMAGES
	//---------------------------------------------------
	cv::Mat currImage, nextImage;
	
	if (1 == _inputImage.channels()) { currImage = _inputImage; }
	else { cv::cvtColor(_inputImage, currImage, CV_BGR2GRAY); }
	
	if (1 == _targetImage.channels()) { nextImage = _targetImage; }
	else { cv::cvtColor(_targetImage, nextImage, CV_BGR2GRAY); }

	//---------------------------------------------------
	// EXTRACT FEATURE POINTS
	//---------------------------------------------------
	if (_vecInputFeatures.empty())
	{
		if (!FeatureExtraction(_inputBox, currImage, _vecInputFeatures))
		{
			return false;
		}
	}

	//---------------------------------------------------
	// FEATURE TRACKING
	//---------------------------------------------------
	std::vector<uchar> vecFeatureStatus;
	_vecOutputFeatures.clear();

	cv::Mat vecErrors;
	cv::calcOpticalFlowPyrLK(
		currImage,
		nextImage,
		_vecInputFeatures,
		_vecOutputFeatures,
		vecFeatureStatus,
		cv::noArray(),
		//vecErrors,
		cv::Size((int)(_inputBox.width * stParam_.dFeatureTrackWindowSizeRatio),
		         (int)(_inputBox.width * stParam_.dFeatureTrackWindowSizeRatio)));

	//---------------------------------------------------
	// BOX ESTIMATION
	//---------------------------------------------------	
	cv::Rect2d newRect = LocalSearchKLT(_inputBox, _vecInputFeatures, _vecOutputFeatures, _vecFeatureInlierIndex);
	if (stParam_.nMinNumFeatures > _vecFeatureInlierIndex.size())
	{ 
		// tracking failure because of insufficient number of tracked feature points
		return false; 
	}
	else
	{
		_trackingResult = newRect;
	}

	return true;
}


/************************************************************************
 Method Name: FindInlierFeatures
 Description:
	- Find inlier feature points
 Input Arguments:
	- _vecInputFeatures : input feature points.
	- _vecOutputFeatures: points of tracking result. Actually it is an output.
	- _vecPointStatus   : tracking status of each point.
 Return Values:
	- std::vector<cv::Point2f>: Inlier points.
************************************************************************/
std::vector<cv::Point2f> CMTTracker::FindInlierFeatures(
	std::vector<cv::Point2f> *_vecInputFeatures,
	std::vector<cv::Point2f> *_vecOutputFeatures,
	std::vector<unsigned char> *_vecPointStatus)
{
	size_t numTrackedFeatures = 0;
	// find center of disparity
	cv::Point2f disparityCenter(0.0f, 0.0f);
	std::vector<cv::Point2f> vecDisparity;
	std::vector<size_t> vecInlierIndex;
	for (size_t pointIdx = 0; pointIdx < _vecPointStatus->size(); pointIdx++)
	{
		if (!(*_vecPointStatus)[pointIdx]) { continue; }
		vecDisparity.push_back((*_vecOutputFeatures)[pointIdx] - (*_vecInputFeatures)[pointIdx]);
		disparityCenter += vecDisparity.back();
		vecInlierIndex.push_back(pointIdx);
		numTrackedFeatures++;
	}
	disparityCenter = (1 / (float)numTrackedFeatures) * disparityCenter;

	// find distribution of disparity norm
	float norm;
	float normAverage = 0.0f;
	float normSqauredAverage = 0.0f;
	float normStd = 0.0;
	std::vector<float> vecNorm;
	for (size_t pointIdx = 0; pointIdx < vecDisparity.size(); pointIdx++)
	{
		norm = (float)cv::norm(vecDisparity[pointIdx] - disparityCenter);
		vecNorm.push_back(norm);
		normAverage += norm;
		normSqauredAverage += norm * norm;
	}
	normAverage /= (float)numTrackedFeatures;
	normSqauredAverage /= (float)numTrackedFeatures;
	normStd = sqrtf(((float)numTrackedFeatures / ((float)numTrackedFeatures - 1)) * (normSqauredAverage - (normAverage * normAverage)));

	std::vector<cv::Point2f> vecInlierFeatures;
	for (size_t pointIdx = 0; pointIdx < vecNorm.size(); pointIdx++)
	{
		if (abs(vecNorm[pointIdx] - normAverage) > 1 * normStd) { continue; }
		vecInlierFeatures.push_back((*_vecOutputFeatures)[vecInlierIndex[pointIdx]]);
	}

	return vecInlierFeatures;
}


/************************************************************************
 Method Name: LocalSearchKLT
 Description:
	- estimate current box location with feature tracking result
 Input Arguments:
	- _preFeatures       : feature positions at the previous frame
	- _curFeatures       : feature positions at the current frame
	- _inlierFeatureIndex: (output) indicates inlier features
 Return Values:
	- cv::Rect: estimated box
************************************************************************/
#define PSN_LOCAL_SEARCH_PORTION_INLIER (false)
#define PSN_LOCAL_SEARCH_MINIMUM_MOVEMENT (0.1)
#define PSN_LOCAL_SEARCH_NEIGHBOR_WINDOW_SIZE_RATIO (0.2)
cv::Rect CMTTracker::LocalSearchKLT(
	cv::Rect _preBox,
	std::vector<cv::Point2f> &_preFeatures,
	std::vector<cv::Point2f> &_curFeatures,
	std::vector<int> &_inlierFeatureIndex)
{
	size_t numFeatures = _preFeatures.size();
	size_t numMovingFeatures = 0;
	_inlierFeatureIndex.clear();
	_inlierFeatureIndex.reserve(numFeatures);

	// find disparity of moving features
	std::vector<cv::Point2d> vecMovingVector;
	std::vector<int> vecMovingFeatuerIdx;
	std::vector<double> vecDx;
	std::vector<double> vecDy;
	vecMovingVector.reserve(numFeatures);
	vecMovingFeatuerIdx.reserve(numFeatures);
	vecDx.reserve(numFeatures);
	vecDy.reserve(numFeatures);
	cv::Point2d movingVector;
	double disparity = 0.0;
	for (int featureIdx = 0; featureIdx < (int)numFeatures; featureIdx++)
	{
		movingVector = _curFeatures[featureIdx] - _preFeatures[featureIdx];
		disparity = hj::NormL2(movingVector);
		if (disparity < PSN_LOCAL_SEARCH_MINIMUM_MOVEMENT * stParam_.dImageRescale) { continue; }

		vecMovingVector.push_back(movingVector);
		vecMovingFeatuerIdx.push_back(featureIdx);
		vecDx.push_back(movingVector.x);
		vecDy.push_back(movingVector.y);

		numMovingFeatures++;
	}

	// check static movement
	if (numMovingFeatures < numFeatures * 0.5)
	{ 
		for (int featureIdx = 0; featureIdx < (int)numFeatures; featureIdx++)
		{
			if (_preBox.contains(_curFeatures[featureIdx]))
			{
				_inlierFeatureIndex.push_back(featureIdx);
			}
		}
		return _preBox;
	}

	std::sort(vecDx.begin(), vecDx.end());
	std::sort(vecDy.begin(), vecDy.end());

	// estimate major disparity
	double windowSize = _preBox.width * PSN_LOCAL_SEARCH_NEIGHBOR_WINDOW_SIZE_RATIO * stParam_.dImageRescale;
	size_t maxNeighborX = 0, maxNeighborY = 0;
	cv::Point2d estimatedDisparity;
	for (size_t disparityIdx = 0; disparityIdx < numMovingFeatures; disparityIdx++)
	{
		size_t numNeighborX = 0;
		size_t numNeighborY = 0;
		// find neighbors in each axis
		for (size_t compIdx = 0; compIdx < numMovingFeatures; compIdx++)
		{
			if (std::abs(vecDx[disparityIdx] - vecDx[compIdx]) < windowSize) { numNeighborX++; } // X		
			if (std::abs(vecDy[disparityIdx] - vecDy[compIdx]) < windowSize) { numNeighborY++; } // Y
		}
		// disparity in X axis
		if (maxNeighborX < numNeighborX)
		{
			estimatedDisparity.x = vecDx[disparityIdx];
			maxNeighborX = numNeighborX;
		}
		// disparity in Y axis
		if (maxNeighborY < numNeighborY)
		{
			estimatedDisparity.y = vecDy[disparityIdx];
			maxNeighborY = numNeighborY;
		}
	}

	// find inliers
	for (int vectorIdx = 0; vectorIdx < (int)numMovingFeatures; vectorIdx++)
	{
		if (hj::NormL2(vecMovingVector[vectorIdx] - estimatedDisparity) < windowSize)
		{
			_inlierFeatureIndex.push_back(vecMovingFeatuerIdx[vectorIdx]);
		}
	}

	// estimate box
	cv::Rect2d estimatedBox = _preBox;
	estimatedBox.x += estimatedDisparity.x;
	estimatedBox.y += estimatedDisparity.y;

	return estimatedBox;
}


/************************************************************************
 Method Name: BoxCenterDistanceWRTScale
 Description:
	- Calculate the distance between two boxes
 Input Arguments:
	- _box1: the first box
	- _box2: the second box
 Return Values:
	- double: distance between two boxes
************************************************************************/
double CMTTracker::BoxCenterDistanceWRTScale(cv::Rect2d &_box1, cv::Rect2d &_box2)
{
	double nominator = hj::NormL2(hj::Center(_box1) - hj::Center(_box2));
	double denominator = (_box1.width + _box2.width) / 2.0;
	double boxDistance = (nominator * nominator) / (denominator * denominator);

	return boxDistance;
}


/************************************************************************
 Method Name: GetTrackingConfidence
 Description:
	- Calculate the confidence of tracking by the number of features lay in the box
 Input Arguments:
	- _box: target position
	- _vecTrackedFeatures: tracked features
 Return Values:
	- tracking confidence
************************************************************************/
double CMTTracker::GetTrackingConfidence(
	cv::Rect &_box, 
	std::vector<cv::Point2f> &_vecTrackedFeatures)
{
	double numFeaturesInBox = 0.0;
	for (std::vector<cv::Point2f>::iterator featureIter = _vecTrackedFeatures.begin();
		featureIter != _vecTrackedFeatures.end();
		featureIter++)
	{
		if (_box.contains(*featureIter))
		{
			numFeaturesInBox++;
		}
	}

	return numFeaturesInBox / (double)_vecTrackedFeatures.size();
}


/************************************************************************
Method Name: GetEstimatedDepth
Description:
	- Estimate the depth of head by averaging depth values in ceter region.
Input Arguments:
	- _frameImage: Entire current input frame image.
	- _objectBox : Head box.
Return Values:
	- double: Estimated detph of the head.
************************************************************************/
double CMTTracker::GetEstimatedDepth(const cv::Mat _frameImage, const cv::Rect _objectBox)
{
	// center region
	int xMin = MAX(0, (int)(_objectBox.x + 0.5 * (1 - stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.width)),
		xMax = MIN(_frameImage.cols, (int)(_objectBox.x + 0.5 * (1 + stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.width) - 1),
		yMin = MAX(0, (int)(_objectBox.y + 0.5 * (1 - stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.height)),
		yMax = MIN(_frameImage.rows, (int)(_objectBox.y + 0.5 * (1 + stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.height) - 1);

	std::vector<uchar> vecCenterDepths;
	vecCenterDepths.reserve((int)(_objectBox.width * _objectBox.height));
	for (int r = yMin; r <= yMax; r++)
	{
		for (int c = xMin; c <= xMax; c++)
		{
			vecCenterDepths.push_back(_frameImage.at<uchar>(r, c));
		}
	}

	// histogram
	std::sort(vecCenterDepths.begin(), vecCenterDepths.end());
	uchar minDepth = vecCenterDepths.front();
	uchar maxDepth = vecCenterDepths.back();
	std::vector<std::pair<uchar, int>> histDepths(maxDepth - minDepth + 1);
	for (size_t i = 0; i < histDepths.size(); i++)
	{
		histDepths[i].first = minDepth + (uchar)i;
		histDepths[i].second = 0;
	}
	for (size_t i = 0; i < vecCenterDepths.size(); i++)
	{
		histDepths[vecCenterDepths[i]-minDepth].second++;
	}

	if (51 == this->nCurrentFrameIdx_)
	{
		int a = 0;
	}

	// count inliers 
	int maxNumInliers = 0;
	uchar estimatedCenterDepth = 0;
	for (int i = 0; i < histDepths.size(); i++)
	{
		int neighborStart = MAX(0, i - (int)(0.5 * stParam_.dDepthForegroundWindowSize));
		int neighborEnd = MIN((int)histDepths.size(), i + (int)(0.5 * stParam_.dDepthForegroundWindowSize)) - 1;
		int curNumInliers = 0;
		for (int j = neighborStart; j <= neighborEnd; j++)
		{
			curNumInliers += histDepths[j].second;
		}
		if (curNumInliers > maxNumInliers)
		{
			maxNumInliers = curNumInliers;
			estimatedCenterDepth = histDepths[i].first;
		}
	}

	return (double)estimatedCenterDepth;
}


/************************************************************************
 Method Name: GetObjectInfo
 Description:
	- Generate current frame's object info from the trajectory.
 Input Arguments:
	- _curTrajectory: Target trajectory.
 Return Values:
	- CObjectInfo: Current frame's state of the target trajectory.
************************************************************************/
CObjectInfo CMTTracker::GetObjectInfo(CTrajectory *_curTrajectory)
{
	CObjectInfo outObjectInfo;
	assert(_curTrajectory->timeEnd == nCurrentFrameIdx_);

	//cv::Rect curBox = hj::Rescale(_curTrajectory->boxes.back(), stParam_.dImageRescaleRecover);
	outObjectInfo.id = _curTrajectory->id;
	outObjectInfo.box = _curTrajectory->boxes.back();
	outObjectInfo.headBox = _curTrajectory->headBoxes.back();
	outObjectInfo.keyPoint = _curTrajectory->keyPoints.back();
	
	return outObjectInfo;
}


/************************************************************************
 Method Name: VisualizeResult
 Description:
	- Visualize the tracking result on the input image frame.
 Input Arguments:
	- None
 Return Values:
	- None
************************************************************************/
void CMTTracker::VisualizeResult()
{
	/* frame information */
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "%04d", this->nCurrentFrameIdx_);
	cv::rectangle(matTrackingResult_, cv::Rect(5, 2, 100, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(matTrackingResult_, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));

	/* detections */
	for (int k = 0; k < vecKeypoints_.size(); k++)
	{
		cv::rectangle(
			matTrackingResult_, 
			hj::Rescale(vecKeypoints_[k].bbox, stParam_.dImageRescaleRecover),
			cv::Scalar(255, 255, 255), 
			1);
	}

	/* tracklets */
	for (int tIdx = 0; tIdx < queueActiveTracklets_.size(); tIdx++)
	{
		CTracklet *curTracklet = queueActiveTracklets_[tIdx];

		//// feature points
		//for (int pointIdx = 0; pointIdx < curTracklet->featurePointsHistory.back().size(); pointIdx++)
		//{
		//	if (pointIdx < curTracklet->trackedPoints.size())
		//	{
		//		cv::circle(
		//			matTrackingResult_,
		//			curTracklet->featurePoints[pointIdx] * stParam_.dImageRescaleRecover,
		//			1, cv::Scalar(0, 255, 0), 1);
		//		cv::line(
		//			matTrackingResult_,
		//			curTracklet->featurePoints[pointIdx] * stParam_.dImageRescaleRecover,
		//			curTracklet->trackedPoints[pointIdx] * stParam_.dImageRescaleRecover,
		//			cv::Scalar(255, 255, 255), 1);
		//		cv::circle(
		//			matTrackingResult_,
		//			curTracklet->trackedPoints[pointIdx] * stParam_.dImageRescaleRecover,
		//			1, cv::Scalar(0, 255, 0), 1);
		//	}
		//	else
		//	{
		//		cv::circle(
		//			matTrackingResult_,
		//			curTracklet->featurePoints[pointIdx] * stParam_.dImageRescaleRecover,
		//			1, cv::Scalar(0, 0, 255), 1);
		//	}
		//}

		// tracklet box
		//hj::DrawBoxWithID(matTrackingResult_, curObject->box, curObject->id, 0, 0, &vecColors_);
	}

	///* trajectories */
	//queueTrajectories base visualize
	//for (int trajIdx = 0; trajIdx < queueActiveTrajectories_.size(); trajIdx++)
	//{
	//	CTrajectory *curTrajectory = queueActiveTrajectories_[trajIdx];

	//	if (curTrajectory->timeEnd != this->nCurrentFrameIdx_) { continue; }
	//	DrawBoxWithID(
	//		matTrackingResult_, 
	//		hj::Rescale(curTrajectory->boxes.back(), stParam_.dImageRescaleRecover), 
	//		curTrajectory->id, 
	//		0, 
	//		0,
	//		getColorByID(curTrajectory->id, &vecColors_));
	//}

	//activateTRajectories based visualize
	for (int trajIdx = 0; trajIdx < activeTrajectories_.size(); trajIdx++)
	{
		CTrajectory *curTrajectory = activeTrajectories_[trajIdx];

		if (curTrajectory->timeEnd != this->nCurrentFrameIdx_) { continue; }
		
		DrawBoxWithID(
			matTrackingResult_,
			hj::Rescale(curTrajectory->boxes.back(), stParam_.dImageRescaleRecover),
			curTrajectory->id,
			0,
			0,
			getColorByID(curTrajectory->id, &vecColors_));

		cv::rectangle(
			matTrackingResult_,
			curTrajectory->headBoxes.back(),
			cv::Scalar(0, 0, 255), 1);

	}


	//---------------------------------------------------
	// RECORD
	//---------------------------------------------------
	if (bVideoWriterInit_)
	{
		IplImage *currentFrame = new IplImage(matTrackingResult_);
		cvWriteFrame(videoWriter_, currentFrame);
		delete currentFrame;
	}

	cv::namedWindow(strVisWindowName_);

	cv::imshow(strVisWindowName_, matTrackingResult_);
	cv::waitKey(1);
	matTrackingResult_.release();
}


cv::Scalar CMTTracker::hsv2rgb(double h, double s, double v)
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


std::vector<cv::Scalar> CMTTracker::GenerateColors(unsigned int numColor)
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


cv::Scalar CMTTracker::getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors)
{
	if (NULL == vecColors) { return cv::Scalar(255, 255, 255); }
	unsigned int colorIdx = nID % vecColors->size();
	return (*vecColors)[colorIdx];
}


void CMTTracker::DrawBoxWithID(
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
		cv::rectangle(imageFrame, box, curColor, 1);
		cv::rectangle(imageFrame, cv::Rect((int)box.x, (int)box.y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y - 1), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
	}
	else
	{
		cv::rectangle(imageFrame, box, curColor, 1 + lineStyle);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, curColor);
	}
}

void CMTTracker::EstimateHeads(KeyPointsSet& _vecCurKeyPoints)
{
	std::vector<cv::Rect2d> vecHeadBox_;

	for (KeyPointsSet::iterator iter = _vecCurKeyPoints.begin(); iter != _vecCurKeyPoints.end();)
	{
		if ((*iter).bbox.height < stParam_.nMinBoxHeight 
			|| (*iter).bbox.width < stParam_.nMinBoxWidth)  // box size validation
		{
			iter = _vecCurKeyPoints.erase(iter);
			continue;
		}

		(*iter).headPoint = cv::Point2d(0.0, 0.0);
		/*Estimate Head Point*/
		if ((*iter).points.at(NOSE).confidence == 0)
		{
			if ((*iter).points.at(REYE).confidence == 0 && (*iter).points.at(LEYE).confidence == 0
				&& (*iter).points.at(REAR).confidence == 0 && (*iter).points.at(LEAR).confidence == 0)
			{
				iter = _vecCurKeyPoints.erase((iter));
				continue;
			}

			int num = 0;
			for (int k = REYE; k <= LEAR; k++)
			{
				if ((*iter).points.at(k).confidence == 0) { continue; }
				(*iter).headPoint.x += (*iter).points.at(k).x;
				(*iter).headPoint.y += (*iter).points.at(k).y;
				num++;
			}
			(*iter).headPoint.x /= num;
			(*iter).headPoint.y /= num;
		}
		else 
		{
			(*iter).headPoint = cv::Point2d((*iter).points.at(NOSE).x, (*iter).points.at(NOSE).y);
		}
		
		/* Head Box Estimate */
		//Calc nose to neck point
		double distance = 0;
		int idx = 0;
		cv::Point2d diff, topLeft, bottomRight;

		if ((*iter).points.at(NECK).confidence == 0)
		{
			iter = _vecCurKeyPoints.erase(iter);
			continue;
		}

		diff = (*iter).headPoint - cv::Point2d((*iter).points.at(NECK).x, (*iter).points.at(NECK).y);
		distance = cv::norm(diff);
		//double max_distance = 20;                                                    // 이렇게 patch에 min 혹은 max size를 정해주는것이 맞는가?
		double min_distance = 15;
		distance = distance < min_distance ? min_distance : distance;

/*
		if ((*iter).points.at(NOSE).confidence == 0)
		{
			diff = head - cv::Point2d((*iter).points.at(NECK).x, (*iter).points.at(NECK).y);
			distance += cv::norm(diff);
			idx++;
		}
		
		if ((*iter).points.at(REAR).confidence != 0 && (*iter).points.at(LEAR).confidence != 0)
		{
			diff = cv::Point2d((*iter).points.at(REAR).x, (*iter).points.at(REAR).y) -
				cv::Point2d((*iter).points.at(LEAR).x, (*iter).points.at(LEAR).y);
			distance += cv::norm(diff) / 2;
			idx++;
		}

		if (idx == 0)
		{
			iter = _vecCurKeyPoints.erase(iter);
			continue;
		}
		distance /= idx;
*/
		topLeft = cv::Point2d((*iter).headPoint.x + distance, (*iter).headPoint.y - distance);
		bottomRight = cv::Point2d((*iter).headPoint.x - distance, (*iter).headPoint.y + distance);	
		(*iter).headBox = cv::Rect2d(topLeft, bottomRight);

		iter++;
	}
}


void CMTTracker::UpdateTrajectories(
	KeyPointsSet _vecCurKeyPoints,                   
	std::deque<CTrajectory*> _activeTrajectories,
	std::deque<CTrajectory*> _inactiveTrajectories)
{
	//active Trajectories update
	std::deque<CTrajectory*> newActiveTrajectories;
	std::deque<CTrajectory*> newInactiveTrajectories;


	//---------------------------------------------------
	// MATCHING STEP 01: active trajectories <-> keypoints
	//---------------------------------------------------

	// calc cost: keypoints <-> active trajectories
	std::vector<float> arrKeyPointToAtiveMatchingCost_(
		_vecCurKeyPoints.size() * _activeTrajectories.size(),
		std::numeric_limits<float>::infinity());
	for (int trajIdx = 0; trajIdx != _activeTrajectories.size(); trajIdx++)
	{
		for (int pointIdx = 0, costPos = trajIdx;
			pointIdx != _vecCurKeyPoints.size();
			pointIdx++, costPos += _activeTrajectories.size())
		{
			double curCost = 0.0;

			// TODO: translation + depth distance
			cv::Point2d diff = cv::Point2d((_activeTrajectories[trajIdx]->latestHeadPoint().x - _vecCurKeyPoints[pointIdx].headPoint.x),
				(_activeTrajectories[trajIdx]->latestHeadPoint().y - _vecCurKeyPoints[pointIdx].headPoint.y));
			double distance = cv::norm(diff);
			

			if (distance > _activeTrajectories[trajIdx]->boxes.back().height)   //width height 말고 고민해서 바꿔야 할듯. (앞사람 뒷사람의 보정도 생각해야한다.)            
				continue;

			if (!hj::CheckOverlap(_activeTrajectories[trajIdx]->headBoxes.back(), _vecCurKeyPoints[pointIdx].headBox))  //overlap안되면 연결 끊기.
				continue;
			curCost += distance;

			arrKeyPointToAtiveMatchingCost_[costPos] = (float)curCost;
		}
	}

	// handling infinite in the cost array
	float maxCost = -1000.0f;
	for (int costIdx = 0; costIdx < arrKeyPointToAtiveMatchingCost_.size(); costIdx++)
	{
		if (!_finitef(arrKeyPointToAtiveMatchingCost_[costIdx]))
			continue;
		if (maxCost < arrKeyPointToAtiveMatchingCost_[costIdx])
			maxCost = arrKeyPointToAtiveMatchingCost_[costIdx];
	}
	maxCost = maxCost + 100.0f;
	for (int costIdx = 0; costIdx < arrKeyPointToAtiveMatchingCost_.size(); costIdx++)
	{
		if (_finitef(arrKeyPointToAtiveMatchingCost_[costIdx]))
			continue;
		arrKeyPointToAtiveMatchingCost_[costIdx] = maxCost;
	}


	// matching & validation
	CHungarianMethod cHungarianMatcher;
	cHungarianMatcher.Initialize(arrKeyPointToAtiveMatchingCost_, (unsigned int)_vecCurKeyPoints.size(), (unsigned int)_activeTrajectories.size());
	stMatchInfo* curMatchInfo = cHungarianMatcher.Match();
	std::vector<bool> vecKeypointMatchedWithTracklet(_vecCurKeyPoints.size(), false);

	for (size_t matchIdx = 0; matchIdx < curMatchInfo->rows.size(); matchIdx++)
	{
		if (maxCost == curMatchInfo->matchCosts[matchIdx]) { continue; }

		CKeyPoints *curKeyPoint = &_vecCurKeyPoints[curMatchInfo->rows[matchIdx]];
		CTrajectory *curTrajectory = _activeTrajectories[curMatchInfo->cols[matchIdx]];

		// updata matched trajectory
		curTrajectory->timeEnd = this->nCurrentFrameIdx_;
		curTrajectory->timeLastUpdate = curTrajectory->timeEnd;
		curTrajectory->duration = curTrajectory->timeEnd - curTrajectory->timeStart + 1;
		curTrajectory->boxes.push_back(curKeyPoint->bbox);
		curTrajectory->headBoxes.push_back(curKeyPoint->headBox);
		curTrajectory->headPoint.push_back(curKeyPoint->headPoint);
		curTrajectory->keyPoints.push_back(curKeyPoint->points);
		
		newActiveTrajectories.push_back(curTrajectory);

		vecKeypointMatchedWithTracklet[curMatchInfo->rows[matchIdx]] = true;
	}
	cHungarianMatcher.Finalize();

	// update matched trajectories
	for (std::deque<CTrajectory*>::iterator iter = _activeTrajectories.begin();
		iter != _activeTrajectories.end();
		iter++)
	{
		if ((*iter)->timeLastUpdate == this->nCurrentFrameIdx_) { continue; }

		newInactiveTrajectories.push_back((*iter));
	}

	//---------------------------------------------------
	// MATCHING STEP 02: inactive trajectories <-> keypoints
	//---------------------------------------------------

	// calc cost: keypoints <-> inactive trajectories
	std::vector<float> arrKeyPointToInactiveMatchingCost_(
		_vecCurKeyPoints.size() * _inactiveTrajectories.size(),
		std::numeric_limits<float>::infinity());
	for (int trajIdx = 0; trajIdx != _inactiveTrajectories.size(); trajIdx++)
	{
		for (int pointIdx = 0, costPos = trajIdx;
			pointIdx != _vecCurKeyPoints.size();
			pointIdx++, costPos += _inactiveTrajectories.size())
		{
			double curCost = 0.0;
			// TODO: translation + depth distance
			cv::Point2d diff = cv::Point2d((_inactiveTrajectories[trajIdx]->latestHeadPoint().x - _vecCurKeyPoints[pointIdx].headPoint.x),
				(_inactiveTrajectories[trajIdx]->latestHeadPoint().y - _vecCurKeyPoints[pointIdx].headPoint.y));
			double distance = cv::norm(diff);
			

			if (distance > _inactiveTrajectories[trajIdx]->boxes.back().height)        //width height 말고 고민해서 바꿔야 할듯. (앞사람 뒷사람의 보정도 생각해야한다.)
				continue;
			if (!hj::CheckOverlap(_inactiveTrajectories[trajIdx]->headBoxes.back(), _vecCurKeyPoints[pointIdx].headBox)) //overlap안되면 연결 끊기
				continue;
			curCost += distance;

			arrKeyPointToInactiveMatchingCost_[costPos] = (float)curCost;
		}
	}

	// handling infinite in the cost array
	maxCost = -1000.0f;
	for (int costIdx = 0; costIdx < arrKeyPointToInactiveMatchingCost_.size(); costIdx++)
	{
		if (!_finitef(arrKeyPointToInactiveMatchingCost_[costIdx]))
			continue;
		if (maxCost < arrKeyPointToInactiveMatchingCost_[costIdx])
			maxCost = arrKeyPointToInactiveMatchingCost_[costIdx];
	}
	maxCost = maxCost + 100.0f;
	for (int costIdx = 0; costIdx < arrKeyPointToInactiveMatchingCost_.size(); costIdx++)
	{
		if (_finitef(arrKeyPointToInactiveMatchingCost_[costIdx]))
			continue;
		arrKeyPointToInactiveMatchingCost_[costIdx] = maxCost;
	}

	// matching & validation( 아래 update matched trajectories부분과 함께 수정이 필요하다)
	CHungarianMethod cInactiveHungarianMatcher;
	cInactiveHungarianMatcher.Initialize(arrKeyPointToInactiveMatchingCost_, (unsigned int)_vecCurKeyPoints.size(), (unsigned int)_inactiveTrajectories.size());
	stMatchInfo* inactiveMatchInfo = cInactiveHungarianMatcher.Match();

	for (size_t matchIdx = 0; matchIdx < inactiveMatchInfo->rows.size(); matchIdx++)
	{
		if (vecKeypointMatchedWithTracklet[inactiveMatchInfo->rows[matchIdx]]) { continue; }
		if (maxCost == inactiveMatchInfo->matchCosts[matchIdx]) { continue; }
		
		CKeyPoints *curKeyPoint = &_vecCurKeyPoints[inactiveMatchInfo->rows[matchIdx]];
		CTrajectory *curTrajectory = _inactiveTrajectories[inactiveMatchInfo->cols[matchIdx]];
		//validation

		// updata matched trajectory
		curTrajectory->timeEnd = this->nCurrentFrameIdx_;
		curTrajectory->timeLastUpdate = curTrajectory->timeEnd;
		curTrajectory->duration = curTrajectory->timeEnd - curTrajectory->timeStart + 1;
		
		curTrajectory->boxes.push_back(curKeyPoint->bbox);
		curTrajectory->headBoxes.push_back(curKeyPoint->headBox);
		curTrajectory->headPoint.push_back(curKeyPoint->headPoint);
		curTrajectory->keyPoints.push_back(curKeyPoint->points);

		newActiveTrajectories.push_back(curTrajectory);
		vecKeypointMatchedWithTracklet[inactiveMatchInfo->rows[matchIdx]] = true;
	}
	cInactiveHungarianMatcher.Finalize();

	// update matched trajectories
	for (std::deque<CTrajectory*>::iterator iter = _inactiveTrajectories.begin();
		iter != _inactiveTrajectories.end();
		iter++)
	{
		if ((*iter)->timeLastUpdate == this->nCurrentFrameIdx_) { continue; }

		newInactiveTrajectories.push_back((*iter));
	}


	//---------------------------------------------------
	// TRAJECTORY GENERATION
	//---------------------------------------------------	
	for (int pointIdx = 0; pointIdx != _vecCurKeyPoints.size(); pointIdx++)
	{
		if (vecKeypointMatchedWithTracklet[pointIdx]) { continue; }
		// generate new trajectories
		// insert to the list
		// get the pointer of just interted trajectory and put that in the active trajectory queue
		CTrajectory newTrajectory;
		newTrajectory.id = this->nNewTrajectoryID_++;
		newTrajectory.timeStart = this->nCurrentFrameIdx_;
		newTrajectory.timeEnd = this->nCurrentFrameIdx_;
		newTrajectory.timeLastUpdate = this->nCurrentFrameIdx_;
		newTrajectory.duration = 1;
		newTrajectory.boxes.push_back(_vecCurKeyPoints[pointIdx].bbox);
		newTrajectory.headBoxes.push_back(_vecCurKeyPoints[pointIdx].headBox);
		newTrajectory.headPoint.push_back(_vecCurKeyPoints[pointIdx].headPoint);
		newTrajectory.keyPoints.push_back(_vecCurKeyPoints[pointIdx].points);

		// generate trajectory instance
		this->listCTrajectories_.push_back(newTrajectory);
		newActiveTrajectories.push_back(&this->listCTrajectories_.back());
	}

	//---------------------------------------------------
	// TRAJECTORY TERMINATION
	//---------------------------------------------------
	//terminate trajectories (refresh inactivated trajectory queue)
	for (std::deque<CTrajectory*>::iterator trajIter = newInactiveTrajectories.begin(); trajIter != newInactiveTrajectories.end();)
	{
		if ((*trajIter)->timeLastUpdate + stParam_.nMaxPendingTime < (int)this->nCurrentFrameIdx_)
		{
			trajIter = newInactiveTrajectories.erase(trajIter);
			continue;
		}
		trajIter++;
	}

	activeTrajectories_ = newActiveTrajectories;
	inactiveTrajectories_ = newInactiveTrajectories;

}

}

//()()
//('')HAANJU.YOO