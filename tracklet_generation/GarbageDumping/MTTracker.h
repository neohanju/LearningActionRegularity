/**************************************************************************
* Title        : CSCTracker
* Author       : Haanju Yoo
* Initial Date : 2014.03.01 (ver. 0.9)
* Version Num. : 1.0 (since 2016.09.16)
* Description  :
*	Single camera multiple target tracker
**************************************************************************/

#pragma once

#include <vector>
#include <queue>
#include <list>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>  // for video recording
#include "haanju_utils.hpp"

namespace hj
{
	
/////////////////////////////////////////////////////////////////////////
// ALGORITHM PARAMETERS
/////////////////////////////////////////////////////////////////////////
struct stParamTrack
{
	//------------------------------------------------
	// METHODS
	//------------------------------------------------
	stParamTrack()
		: nImageWidth(0)
		, nImageHeight(0)
		, dImageRescale(1.0)
		, dImageRescaleRecover(1.0)
		, dDepthEstimateCenterRegionRatio(0.4)
		, dDepthForegroundWindowSize(30)
		, nMaxTrackletLength(5)
		, nMinNumFeatures(4)
		, nMaxNumFeatures(100)
		, nBackTrackingLength(4)
		, dFeatureTrackWindowSizeRatio(1.0)
		, dMaxBoxDistance(1.0)
		, dMinBoxOverlapRatio(0.3)
		, dMaxBoxCenterDiffRatio(0.5)
		, dMinOpticalFlowMajorityRatio(0.5)
		, dMaxTranslationDistance(30.0)
		, dMaxDepthDistance(30.0)
		, nMaxPendingTime(100) 
		, nMinBoxHeight(100)
		, nMinBoxWidth(20)
		, bVisualize(false)
		, bVideoRecord(false)
		, strVideoRecordPath("")
	{};
	~stParamTrack() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
	int nImageWidth;
	int nImageHeight;

	/* speed-up */
	double dImageRescale;          // Image rescaling factor for speed-up.
	double dImageRescaleRecover;   // For restore image scale. Computed automatically with 'dImageRescale' variable. So, do not manually set this value.

	/* depth estimation */
	double dDepthEstimateCenterRegionRatio;
	double dDepthForegroundWindowSize;

	/* bi-directional tracking */
	int    nMaxTrackletLength;     // To cut off tracklets that are too long and unreliable.
	int    nMinNumFeatures;        // The minimum number of tracked feature points that are required to maintain a tracklet.
	int    nMaxNumFeatures;        // To prevent tracking too many feature points.
	int    nBackTrackingLength;    // The interval of bi-directional tracking of feature points.
	double dFeatureTrackWindowSizeRatio;  // The optical flow searching window size w.r.t. the size of a detection box.
	double dMaxBoxDistance;        // For validation condition
	double dMinBoxOverlapRatio;    // For validation condition
	double dMaxBoxCenterDiffRatio; // For validation condition
	double dMinOpticalFlowMajorityRatio;  // To filter out an ambiguities in the ownership of each feature point.

	/* matching related */
	double dMaxTranslationDistance;
	double dMaxDepthDistance;
	int    nMaxPendingTime;
	int    nMinBoxHeight;
	int    nMinBoxWidth;

	/* visualization for debugging */
	bool bVisualize;

	/* video recording for result visualization */
	bool        bVideoRecord;
	std::string strVideoRecordPath;
};


/////////////////////////////////////////////////////////////////////////
// MULTI-TARGET TRACKER
/////////////////////////////////////////////////////////////////////////
class CMTTracker
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CMTTracker();
	~CMTTracker();

	void Initialize(stParamTrack &_stParams);
	void Finalize(void);
	CTrackResult Track(
		KeyPointsSet _vecInputDetections,
		cv::Mat _curFrame, 
		int _frameIdx);

private:
	/* MAIN OPERATIONS */	
	//void GenerateDetectedObjects(
	//	const cv::Mat _frameImage,
	//	KeyPointsSet &_vecDetections,
	//	std::vector<CDetectedObject> &_vecDetectedObjects);
	TrackletPtQueue UpdateTracklets(
		TrackletPtQueue _keyPointsTracklets,
		TrackletPtQueue _activeTracklets);
	//void DetectionToTrackletMatching(
	//	const std::vector<CDetectedObject> &_vecDetectedObjects, 
	//	TrackletPtQueue &_queueTracklets);
	void TrackletToTrajectoryMatching(const TrackletPtQueue &_queueActiveTracklets);
	void ResultPackaging();

	/* TRACKING RELATED */
	bool FeatureExtraction(
		const cv::Rect2d _inputBox, 
		const cv::Mat _inputImage, 
		std::vector<cv::Point2f> &_vecFeaturePoints);
	bool FeatureTracking(
		const cv::Rect2d _inputBox, 
		const cv::Mat _inputImage, 
		const cv::Mat _targetImage, 
		std::vector<cv::Point2f> &_vecInputFeatures, 
		std::vector<cv::Point2f> &_vecOutputFeatures, 
		std::vector<int> &_vecFeatureInlierIndex, 
		cv::Rect2d &_trackingResult);
	std::vector<cv::Point2f> FindInlierFeatures(
		std::vector<cv::Point2f> *_vecInputFeatures, 
		std::vector<cv::Point2f> *_vecOutputFeatures, 
		std::vector<unsigned char> *_vecPointStatus);
	cv::Rect LocalSearchKLT(
		cv::Rect _preBox, 
		std::vector<cv::Point2f> &_preFeatures, 
		std::vector<cv::Point2f> &_curFeatures, 
		std::vector<int> &_inlierFeatureIndex);
	static double BoxCenterDistanceWRTScale(cv::Rect2d &_box1, cv::Rect2d &_box2);
	static double GetTrackingConfidence(cv::Rect &_box, std::vector<cv::Point2f> &_vecTrackedFeatures);

	/* ETC */
	double GetEstimatedDepth(const cv::Mat _frameImage, const cv::Rect _objectBox);
	CObjectInfo GetObjectInfo(CTrajectory *_curTrajectory);

	/* USING ONLY HEAD INFO */
	void EstimateHeads(KeyPointsSet& _vecCurKeyPoints);
	void UpdateTrajectories(
		const KeyPointsSet _vecCurKeyPoints,
		std::deque<CTrajectory*> _activeTrajectories,
		std::deque<CTrajectory*> _inactiveTrajectories);

	/* VISUALIZATION */
	void VisualizeResult();
	cv::Scalar hsv2rgb(double h, double s, double v);
	std::vector<cv::Scalar> GenerateColors(unsigned int numColor);
	cv::Scalar getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors);
	void DrawBoxWithID(cv::Mat &imageFrame, cv::Rect box, unsigned int nID, int lineStyle, int fontSize, cv::Scalar curColor);

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	bool         bInit_;
	stParamTrack stParam_;	
	unsigned int nCurrentFrameIdx_;

	/* calibration related */	
	unsigned int nInputWidth_;
	unsigned int nInputHeight_;	

	/* input related */
	//std::vector<CDetectedObject> vecDetectedObjects_;
	KeyPointsSet vecKeypoints_;
	hj::CMatFIFOBuffer cImageBuffer_;
	cv::Size sizeBufferImage_;
	cv::Mat  matGrayImage_;
	cv::Mat  matResizedGrayImage_;	
	
	/* tracklet related */
	unsigned int         nNewTrackletID_;
	std::list<CTracklet> listCTracklet_;
	TrackletPtQueue      queueActiveTracklets_;
	TrackletPtQueue      queueNewTracklets_;

	/* trajectory related */
	unsigned int             nNewTrajectoryID_;
	std::list<CTrajectory>   listCTrajectories_;
	std::deque<CTrajectory*> queueActiveTrajectories_;
	//head Track related
	std::deque<CTrajectory*> activeTrajectories_;
	std::deque<CTrajectory*> inactiveTrajectories_;

	/* matching related */
	std::vector<float> arrKeyPointToTrackletMatchingCost_;
	std::vector<float> arrInterTrackletMatchingCost_;

	/* feature tracking related */
	cv::Ptr<cv::AgastFeatureDetector> featureDetector_;
	cv::Mat matFeatureExtractionMask_;

	/* result related */
	CTrackResult trackingResult_;

	/* visualization related */
	bool        bVisualizeResult_;
	cv::Mat     matTrackingResult_;
	std::string strVisWindowName_;
	std::vector<cv::Scalar> vecColors_;

	// record
	bool bRecord_;
	bool bVideoWriterInit_;
	std::string strRecordPath_;
	CvVideoWriter *videoWriter_;

	/* head patch related */
	//cv::Mat matHeadPatch_;
	std::vector<cv::Rect2d> vecHeadPatch_;
	
};

///////////////////////////////////////
//Pose KeyPoint access
///////////////////////////////////////

enum KeypointBaseIdx{
	NOSE = 0,
	NECK,
	RSHOULDER,
	RELBOW,
	RWRIST,
	LSHOULDER,
	LELBOW,
	LWRIST,
	RHIP,
	RKNEE,
	RANKLE,
	LHIP,
	LKNEE,
	LANKLE,
	REYE,
	LEYE,
	REAR,
	LEAR,
	BKG
};



}

//()()
//('')HAANJU.YOO
