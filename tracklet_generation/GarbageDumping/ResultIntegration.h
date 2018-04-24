#pragma once
#include <opencv2/highgui/highgui.hpp>

#include "ActionClassifier.h"
#include "ThrowDetectorSet.h"

namespace jm
{

struct stParamResult
{
	stParamResult()
		: bVisualize(true)
	{};

	~stParamResult() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------

	bool bVisualize;
};

struct stDetectResult
{
	stDetectResult() : 
		trackId(0), 
		bActionDetect(false), 
		bKCFDetect(false), 
		bMASKDetect(false) {}

	stDetectResult(unsigned int _trackId, bool _bActionDetect, bool _bKCFDetect, bool _bMASKDetect): 
		trackId(_trackId), 
		bActionDetect(_bActionDetect), 
		bKCFDetect(_bKCFDetect), 
		bMASKDetect(_bMASKDetect){}

	unsigned int trackId;
	bool bActionDetect;
	bool bKCFDetect;
	bool bMASKDetect;

	cv::Rect2d   box;
	cv::Rect2d   headBox;
	std::vector<hj::stKeyPoint> keyPoint;
};

/////////////////////////////////////////////////////////////////////////
// FINAL RESULT (OF ENTIRE TARGETS)
/////////////////////////////////////////////////////////////////////////
class CDetectResultSet
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CDetectResultSet() : frameIdx(0), timeStamp(0), procTime(0) {}
	~CDetectResultSet() {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int frameIdx;
	unsigned int timeStamp;
	bool bThrowDetect;
	time_t procTime;
	std::vector<stDetectResult> detectResults;
};

class CResultIntegration
{
	//----------------------------------------------------------------
	// METHODS
	//---------------------------------------------------------------
public:
	CResultIntegration();
	~CResultIntegration();

	void Initialize(stParamResult &stParams_);
	void Finalize();
	CDetectResultSet Run(hj::CTrackResult *_trackResult, CThrownResultSet *_throwResult, jm::CActionResultSet *_actionResult, cv::Mat _curFrame, int _frameIdx);

private:

	void Integrate(/*hj::CTrackResult *_trackResult, jm::CActionResultSet *_actionResult*/);
	void Visualize();
	unsigned int DrawBoxWithID( cv::Mat &imageFrame, cv::Rect box, unsigned int nID, int lineStyle, int fontSize, cv::Scalar curColor);
	cv::Scalar getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors);
	cv::Scalar hsv2rgb(double h, double s, double v);
	std::vector<cv::Scalar> GenerateColors(unsigned int numColor);
	

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	bool bInit_;
	stParamResult stResultParam_;
	unsigned int     nCurrentFrameIdx_;


	/* visualization related */
	bool             bVisualizeResult_;
	std::string      strVisWindowName_;
	cv::Mat          matResult_;
	std::vector<cv::Scalar> vecColors_;


	CDetectResultSet  integratedResult_;
	hj::CTrackResult  curTrackResult_;
	CActionResultSet  curActionResult_;
	CThrownResultSet curThrowResult_;

	/*Detection relate*/
	bool bSVMResult;
	bool bThrowResult;

};
}