#pragma once

#include <vector>
#include <deque>
#include <opencv2\core.hpp>

#define NUM_KEYPOINT_TYPES (18)
#define KEYPOINTS_BBOX_MARGIN (0.1)

enum TRACKING_DIRECTION { FORWARD = 0, BACKWARD };

namespace hj
{

/////////////////////////////////////////////////////////////////////////
// DETECTION (FOR THE INPUT OF THE TRACKING ALGORITHM & EVALUATION MODULE)
/////////////////////////////////////////////////////////////////////////
struct stKeyPoint
{
	stKeyPoint() : x(0.0), y(0.0), confidence(0.0), cvPoint(x, y) {}
	stKeyPoint(double _x, double _y, double _confidence = 1.0)
		: x(_x), y(_y), confidence(_confidence), cvPoint(_x, _y) {}
	double normL2()
	{
		return std::sqrt(x*x + y*y);
	}
	cv::Point2d cv() const
	{
		return this->cvPoint;
	}
	stKeyPoint operator- (const stKeyPoint _op) { return stKeyPoint(this->x - _op.x, this->y - _op.y); }
	void rescale(double scale_) {
		x *= scale_;
		y *= scale_;
		cvPoint = cv::Point2d(x, y);
	}
	double x;
	double y;
	double confidence;
private:
	cv::Point2d cvPoint;
};

class CKeyPoints
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CKeyPoints() : confidence(0.0) {}
	CKeyPoints(const CKeyPoints &_det) { *this = _det; }
	~CKeyPoints() {}
	void Set(const std::vector<stKeyPoint> _keypoints)
	{
		this->points = _keypoints;
		this->confidence = 0.0;
		double xmin = DBL_MAX, xmax = 0.0, ymin = DBL_MAX, ymax = 0.0;
		for (int i = 0; i < this->points.size(); i++)
		{
			if (0.0 == this->points[i].x && 0.0 == this->points[i].y)
				continue;  // skip undetected keypoint

			xmin = std::min(xmin, this->points[i].x);
			xmax = std::max(xmax, this->points[i].x);
			ymin = std::min(ymin, this->points[i].y);
			ymax = std::max(ymax, this->points[i].y);
			this->confidence += this->points[i].confidence;
		}
		double bboxHeight = (ymax - ymin + 1.0);
		double bboxMargin = bboxHeight * KEYPOINTS_BBOX_MARGIN;
		this->bbox = cv::Rect2d(
			xmin - bboxMargin,
			ymin - bboxMargin,
			xmax - xmin + 1.0 + 2.0 * bboxMargin,
			bboxHeight * (1.0 * 2.0 * bboxMargin));
		this->confidence /= (double)this->points.size();
	}
	CKeyPoints resize(double scale_) 
	{
		CKeyPoints resizedKeyPoints(*this);
		for (int i = 0; i < this->points.size(); ++i) 
		{

		}
	}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	std::vector<stKeyPoint> points;
	cv::Rect2d bbox;  // bounding box
	cv::Rect2d headBox;
	cv::Point2d headPoint;
	double confidence;
	// int nFrame; // Action recognition Related (Temporary implementation)
	// int jsonId; // Action recognition Related (Temporary implementation)
};
typedef std::vector<CKeyPoints> KeyPointsSet;


/////////////////////////////////////////////////////////////////////////
// FOR PER FRAME TRACKING RESULT
/////////////////////////////////////////////////////////////////////////
class CObjectInfo
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CObjectInfo() : id(0), box(0.0, 0.0, 0.0, 0.0) {}
	~CObjectInfo() {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int id;
	cv::Rect2d   box;
	cv::Rect2d   headBox;
	std::vector<hj::stKeyPoint> keyPoint;

};
//
//
///////////////////////////////////////////////////////////////////////////
//// VARIATION OF DETECTION OBJECT FOR BACKWARD FEATURE POINT TRACKING
///////////////////////////////////////////////////////////////////////////
//class CDetectedObject
//{
//	//----------------------------------------------------------------
//	// METHODS
//	//----------------------------------------------------------------
//public:
//	CDetectedObject();
//	~CDetectedObject();
//
//	//----------------------------------------------------------------
//	// VARIABLES
//	//----------------------------------------------------------------
//public:
//	int        id;
//	CKeyPoints keypoints;
//	bool       bMatchedWithTracklet;
//	bool       bCoveredByOtherDetection;
//
//	/* backward tracking related */
//	std::vector<std::vector<cv::Point2f>> vecvecTrackedFeatures; // current -> past order
//	std::vector<cv::Rect2d> boxes; // current -> past order
//};

class CTrajectory;
/////////////////////////////////////////////////////////////////////////
// TRACKLET
/////////////////////////////////////////////////////////////////////////
typedef std::vector<cv::Point2f> cvPoint2fSet;
class CTracklet
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CTracklet();
	~CTracklet();
	int length() { return (int)this->queueKeyPoints.size(); }
	cv::Rect2d currentBox() { assert(queueKeyPoints.size() > 0);  return queueKeyPoints.back().bbox; }
	void insertKeyPoints(const CKeyPoints _keyPoints, int _timeIndex);
	void replaceKeyPoints(const CKeyPoints _keyPoints, int _timeIndex);
	cv::Point2d curHeadPoint() { return this->queueKeyPoints.front().headPoint; }

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	int id;
	int timeStart;
	int timeEnd;
	TRACKING_DIRECTION direction;
	double confidence;
	std::deque<CKeyPoints> queueKeyPoints;
	std::deque<cvPoint2fSet> featurePointsHistory;
	CTrajectory *ptTrajectory;
};
typedef std::deque<CTracklet*> TrackletPtQueue;


/////////////////////////////////////////////////////////////////////////
// TRAJECTORY (FINAL TRACKING RESULT OF EACH TARGET)
/////////////////////////////////////////////////////////////////////////
class CTrajectory
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CTrajectory();
	~CTrajectory();
	cv::Point2d latestHeadPoint() { return this->headPoint.back(); }

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	int id;
	int timeStart;
	int timeEnd;
	int timeLastUpdate;
	int duration;
	double confidence;
	std::deque<cv::Rect2d> boxes;
	std::deque<cv::Rect2d> headBoxes; //jm
	std::deque<cv::Point2d> headPoint;      //jm
	TrackletPtQueue tracklets;
	std::deque<std::vector<stKeyPoint>> keyPoints;  //jm (original: Tracklet)

};
typedef std::deque<CTrajectory> TrajectoryVector;


/////////////////////////////////////////////////////////////////////////
// TRACKING RESULT (OF ENTIRE TARGETS)
/////////////////////////////////////////////////////////////////////////
class CTrackResult
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CTrackResult() : frameIdx(0), timeStamp(0), procTime(0) {}
	~CTrackResult() {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int frameIdx;
	unsigned int timeStamp;
	time_t procTime;
	hj::KeyPointsSet vecKeypoints;  // for the convenience of debugging
	std::vector<CObjectInfo> objectInfos;
	std::vector<cv::Rect> vecDetectionRects;
	std::vector<cv::Rect> vecTrackerRects;
	cv::Mat matMatchingCost;	
};

}

//()()
//('')HAANJU.YOO
