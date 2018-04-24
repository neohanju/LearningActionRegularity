#pragma once
#include <deque>
#include <list>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\ml.hpp>

//#include "haanju_utils.hpp"
#include "MTTracker.h"

namespace jm
{
	/////////////////////////////////////////////////////////////////////////
	// ALGORITHM PARAMETERS
	/////////////////////////////////////////////////////////////////////////
	struct stParamAction
	{
		stParamAction()
			: nPoseLength(30)
			, nMaxPendingFrame(5)
			, nStepSize(5)
			, bTrained(true)
			, bVisualize(true)
			, bNormalize(true)
			, bUsingDisparity(true)
		{};

		~stParamAction() {};

		//------------------------------------------------
		// VARIABLES
		//------------------------------------------------
		int  nPoseLength;
		int  nMaxPendingFrame;
		int  nStepSize;

		bool bTrained;
		bool bVisualize;
		bool bNormalize;
		bool bUsingDisparity;
		
	};


	struct stActionResult
	{
		stActionResult() : trackId(0), bActionDetect(false) {}
		stActionResult(unsigned int _trackId, bool _bActionDetect)
			: trackId(_trackId), bActionDetect(_bActionDetect) {}

		unsigned int trackId;
		bool bActionDetect;

	};

	/////////////////////////////////////////////////////////////////////////
	// POSE CLASSIFICATION RESULT (OF ENTIRE TARGETS)
	/////////////////////////////////////////////////////////////////////////
	class CActionResultSet
	{
		//----------------------------------------------------------------
		// METHODS
		//----------------------------------------------------------------
	public:
		// constructors
		CActionResultSet() : frameIdx(0), timeStamp(0), procTime(0) {}
		~CActionResultSet() {}

		//----------------------------------------------------------------
		// VARIABLES
		//----------------------------------------------------------------
	public:
		unsigned int frameIdx;
		unsigned int timeStamp;
		time_t procTime;
		std::vector<stActionResult> actionResults;      // 이름 다시 생각해보기(겹치는 이름이 있음)
	};

	//typedef std::vector<hj::stKeyPoint> CPosePoints;
	//typedef std::deque<CPosePoints> CAction;
	typedef std::deque<hj::CObjectInfo> CAction;   //change this in final implementation
												   // continuous frame pose 
	class CPoselet
	{
		//----------------------------------------------------------------
		// METHODS
		//---------------------------------------------------------------
	public:
		CPoselet()
			: lastUpdate(-1)
		{};
		~CPoselet() {};
		//void interpolation() {};

		//------------------------------------------------
		// VARIABLES
		//------------------------------------------------
	public:
		int id;
		int nStartFrame;
		int nEndFrame;
		int duration;
		int lastUpdate;      // 30frame씩 묶는것 관련
							 //CAction vectorPose;
		CAction vectorObjInfo;       // 이름 다시 생각해보기
		bool bActionDetect;          // TODO: change CAction

	};

	// typedef std::deque<CAction> ActionSet;

	class CActionClassifier
	{
	public:
		CActionClassifier();
		~CActionClassifier();

		void Initialize(stParamAction &stParam/*, std::string _strModelPath*/);
		void Finalize();
		CActionResultSet Run(hj::CTrackResult *_curTrackResult, cv::Mat _curFrame, int frameIdx);

	private:
		void Detect(std::deque<CPoselet*> _activePoselets, hj::CTrackResult *_curTrackResult);
		void TrainSVM(std::string _saveModelPath);
		void UpdatePoseletUsingTrack();
		void ResultPackaging();
		void Visualize(hj::CTrackResult *_curTrackResult);
		void EliminationStepSize();

		/* Normalize relate function */
		double CalcNormDist(std::vector<hj::stKeyPoint> _curKeypoints);


	public:
		bool             bInit_;
		stParamAction    stActionParam_;
		unsigned int     nCurrentFrameIdx_;
		hj::CTrackResult curTrackResult_;         //있어야 하나? 없어도 될듯!

		/* visualization related */
		bool             bVisualizeResult_;
		cv::Mat          matDetectResult_;
		std::string      strVisWindowName_;

		/* detection input data related */
		std::list<CPoselet>   listCPoselets_;
		std::deque<CPoselet*> pendingPoselets_;
		std::deque<CPoselet*> activePoselets_;

		std::deque<CAction*>  testActions_;
		std::list<CAction>    listCActions_;

		cv::Ptr<cv::ml::SVM> svm;

		/* result related */
		std::vector<stActionResult> listActionResult;
		CActionResultSet actionResult_;         // 이름 다시 생각해보기

	private:

	};

	//
	//class CActionResult
	//{
	//	//----------------------------------------------------------------
	//	// METHODS
	//	//----------------------------------------------------------------
	//
	//public:
	//	CActionResult() : trackId(0), bActionDetect(false) {}
	//	~CActionResult() {}
	//
	//	//----------------------------------------------------------------
	//	// VARIABLES
	//	//----------------------------------------------------------------
	//public:
	//	unsigned int trackId;
	//	bool bActionDetect;
	//
	//};





}
