#include "haanju_types.hpp"

namespace hj
{

///////////////////////////////////////////////////////////////////////////
//// CDetectedObject MEMBER FUNCTIONS
///////////////////////////////////////////////////////////////////////////
//CDetectedObject::CDetectedObject()
//	: id(0)
//	, bMatchedWithTracklet(false)
//	, bCoveredByOtherDetection(false)
//{
//}
//
//
//CDetectedObject::~CDetectedObject()
//{
//	for (size_t vecIdx = 0; vecIdx < vecvecTrackedFeatures.size(); vecIdx++)
//	{
//		vecvecTrackedFeatures[vecIdx].clear();
//	}
//	vecvecTrackedFeatures.clear();
//	boxes.clear();
//}


/////////////////////////////////////////////////////////////////////////
// CTracklet MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CTracklet::CTracklet()
	: id(0)
	, timeStart(0)
	, timeEnd(0)
	, confidence(0.0)
	, ptTrajectory(NULL)
{
}


CTracklet::~CTracklet()
{
}


void CTracklet::insertKeyPoints(const CKeyPoints _keyPoints, int _timeIndex)
{
	if (this->queueKeyPoints.size() > 0)
	{
		// examine the direction of tracking
		if (1 == this->queueKeyPoints.size())
		{
			this->direction = this->timeStart < _timeIndex ? FORWARD : BACKWARD;
		}
		// do not allow to have a gap
		int properTimeIndex = this->direction == FORWARD ? this->timeEnd + 1 : this->timeEnd - 1;
		assert(properTimeIndex == _timeIndex);
	}
	else
	{
		this->timeStart = _timeIndex;
	}
	this->timeEnd = _timeIndex;
	this->confidence = 
		(this->queueKeyPoints.size() * this->confidence + _keyPoints.confidence) 
		/ (double)(this->queueKeyPoints.size() + 1);
	this->queueKeyPoints.push_back(_keyPoints);
}


void CTracklet::replaceKeyPoints(const CKeyPoints _keyPoints, int _timeIndex)
{
	int pos = FORWARD == this->direction ? _timeIndex - timeStart : timeStart - _timeIndex;
	assert(pos < this->length());
	this->queueKeyPoints[pos] = _keyPoints;
}


/////////////////////////////////////////////////////////////////////////
// CTrajectory MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CTrajectory::CTrajectory()
	: id(0)
	, timeStart(0)
	, timeEnd(0)
	, timeLastUpdate(0)
	, duration(0)
{
}


CTrajectory::~CTrajectory()
{
}

}