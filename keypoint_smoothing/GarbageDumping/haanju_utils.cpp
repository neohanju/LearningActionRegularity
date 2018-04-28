#include "haanju_utils.hpp"
#include <opencv2/imgproc/imgproc.hpp>


/************************************************************************
 Method Name: FormattedString
 Description:
	- Printf function for std::string type.
 Input Arguments:
	- _formatted_string: The formatted string input.
	- ...              : The assigning values of '_formatted_string'.
 Return Values:
	- std::string: The result of the formatted string.
************************************************************************/
std::string hj::FormattedString(const std::string _formatted_string, ...)
{
	int final_n, n = ((int)_formatted_string.size()) * 2; /* Reserve two times as much as the length of the _formatted_string */
	std::string str;
	std::unique_ptr<char[]> formatted;
	va_list ap;
	while (1)
	{
		formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
		strcpy_s(formatted.get(), n, _formatted_string.c_str());
		va_start(ap, _formatted_string);
		final_n = vsnprintf(&formatted[0], n, _formatted_string.c_str(), ap);
		va_end(ap);
		if (final_n < 0 || final_n >= n)
		{
			n += abs(final_n - n + 1);
		}
		else
		{
			break;
		}
	}
	return std::string(formatted.get());
}


/////////////////////////////////////////////////////////////////////////
// hj::CMatFIFOBuffer MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
bool hj::CMatFIFOBuffer::set(int _bufferSize)
{
	if (bInit_) { this->clear(); }
	try
	{
		bufferSize_ = _bufferSize;
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::set. Exeption number is %d.\n", e);
		return false;
	}
	return bInit_ = true;
}


bool hj::CMatFIFOBuffer::clear()
{
	if (!bInit_) { return true; }
	try
	{
		for (int bufferIdx = 0; bufferIdx < buffer_.size(); bufferIdx++)
		{
			this->remove(bufferIdx);
		}
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::clear. Exeption number is %d.\n", e);
		return false;
	}
	bufferSize_ = 0;
	bInit_ = false;

	return true;
}


bool hj::CMatFIFOBuffer::insert(cv::Mat _newMat)
{
	try
	{
		cv::Mat newBufferMat = _newMat.clone();
		buffer_.push_back(newBufferMat);

		// circulation
		if (bufferSize_ < buffer_.size())
		{
			buffer_.pop_front();
		}
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::insert. Exeption number is %d.\n", e);
		return false;
	}

	return true;
}


bool hj::CMatFIFOBuffer::insert_resize(cv::Mat _newMat, cv::Size _resizeParam)
{
	try
	{
		cv::Mat newBufferMat;
		cv::resize(_newMat, newBufferMat, _resizeParam);
		buffer_.push_back(newBufferMat);

		// circulation
		if (bufferSize_ < buffer_.size())
		{
			buffer_.pop_front();
		}
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::insert. Exeption number is %d.\n", e);
		return false;
	}

	return true;
}


bool hj::CMatFIFOBuffer::remove(int _pos)
{
	assert(bInit_ && _pos < bufferSize_);
	if (_pos >= buffer_.size())
	{
		return true;
	}
	if (!buffer_[_pos].empty())
	{
		buffer_[_pos].release();
	}
	return true;
}

#include <fstream>
#include <iostream>
const int kClassLabelIdx = 3 + NUM_KEYPOINT_TYPES * 3;
std::vector<hj::KeyPointsSet> hj::ReadKeypoints(const std::string _strFilePath)
{
	std::vector<KeyPointsSet> vecKeyPointsSet;
	try {
		std::ifstream input(_strFilePath.c_str());

		int curID = -1;
		for (std::string line; getline(input, line); )
		{
			int tokenIdx = 0;
			CKeyPoints newKeyPoints;

			std::istringstream ss(line);			
			int elementIdx = 0;
			double x, y, confidence;
			for (std::string token; getline(ss, token, ' '); ++tokenIdx)
			{				
				if (0 == tokenIdx)  // ID
				{
					newKeyPoints.id = std::stoi(token);									 
				}
				else if (1 == tokenIdx)  // suspect(111) or not(1)
				{
					newKeyPoints.isSuspect = (0 == token.compare("111"));
				}
				else if (2 == tokenIdx)  // frame #
				{
					newKeyPoints.frameIndex = std::stoi(token);
				}
				else if (kClassLabelIdx == tokenIdx)
				{
					newKeyPoints.isThrowingGarbage = (0 == token.compare("1"));
				}
				else
				{					
					if (0 == elementIdx)
						x = std::stod(token);
					else if (1 == elementIdx)
						y = std::stod(token);
					else
						confidence = std::stod(token);
					
					if (2 <= elementIdx)
					{
						newKeyPoints.points.push_back(stKeyPoint(x, y, confidence));
						elementIdx = 0;
					}
					else
						elementIdx++;
				}				
			}
			if (curID != newKeyPoints.id)
			{
				curID = newKeyPoints.id;
				vecKeyPointsSet.push_back(KeyPointsSet());
			}
			vecKeyPointsSet.back().push_back(newKeyPoints);
		}		
	}
	catch (int nError) 
	{
		printf("[ERROR] file open error with detection result reading: %d\n", nError);
	}

	return vecKeyPointsSet;
}


bool hj::WriteKeypoints(const std::string _file_path, const std::vector<KeyPointsSet> _vec_key_point_sets)
{
	try {
		std::ofstream output(_file_path.c_str());
		for (int sIdx = 0; sIdx < _vec_key_point_sets.size(); ++sIdx)
		{
			for (int kpIdx = 0; kpIdx < _vec_key_point_sets[sIdx].size(); ++kpIdx)
			{
				std::string isSuspect = _vec_key_point_sets[sIdx][kpIdx].isSuspect ? "111" : "1";				
				output
					<< _vec_key_point_sets[sIdx][kpIdx].id << ' '
					<< isSuspect << ' '
					<< _vec_key_point_sets[sIdx][kpIdx].frameIndex << ' ';
				for (int ptIdx = 0; ptIdx < _vec_key_point_sets[sIdx][kpIdx].points.size(); ++ptIdx)
				{
					output
						<< _vec_key_point_sets[sIdx][kpIdx].points[ptIdx].x << ' '
						<< _vec_key_point_sets[sIdx][kpIdx].points[ptIdx].y << ' '
						<< _vec_key_point_sets[sIdx][kpIdx].points[ptIdx].confidence << ' ';
				}
				output << (int)_vec_key_point_sets[sIdx][kpIdx].isThrowingGarbage << std::endl;
			}
		}
	}
	catch (int nError)
	{
		printf("[ERROR] file open error: %d\n", nError);
		return false;
	}

	return true;
}


double hj::KeyPointsDistance(
	hj::CKeyPoints _firstKeypoints,
	hj::CKeyPoints _secondKeypoints,
	bool _average,
	std::vector<double> *_ptPointWeights)
{
	std::vector<double> pointWeights;
	assert(_firstKeypoints.points.size() == _secondKeypoints.points.size());
	if (NULL != _ptPointWeights)
	{
		assert(_firstKeypoints.points.size() == _ptPointWeights->size());
		pointWeights = *_ptPointWeights;
	}
	else
	{
		pointWeights.resize(_firstKeypoints.points.size(), 1.0);
	}

	double totalDistance = 0.0;
	for (int i = 0; i < pointWeights.size(); i++)
	{
		totalDistance += pointWeights[i] * (_firstKeypoints.points[i] - _secondKeypoints.points[i]).normL2();
	}
	return _average ? totalDistance / (double)pointWeights.size() : totalDistance;
}


bool hj::CheckOverlap(const hj::CKeyPoints _firstKeyPoints, const hj::CKeyPoints _secondKeyPoints)
{
	// examine bounding box
	if (hj::CheckOverlap(_firstKeyPoints.bbox, _secondKeyPoints.bbox))
		return true;

	// examine each point whether the point is at the inside of the other bounding box
	for (int i = 0; i < _firstKeyPoints.points.size(); i++)
	{
		if (_secondKeyPoints.bbox.contains(_firstKeyPoints.points[i].cv()))
			return true;
	}
	for (int i = 0; i < _secondKeyPoints.points.size(); i++)
	{
		if (_firstKeyPoints.bbox.contains(_secondKeyPoints.points[i].cv()))
			return true;
	}
	return false;
}

#include "SGSmoother.h"
const int kMaxTimeGap = 10;
const double kMinConfidence = 0.1;
const bool bSmoothing[NUM_KEYPOINT_TYPES] = {
	true,  // 0
	true,  // 1
	true,  // 2
	false,
	false,
	true,  // 5
	false,
	false,
	true,  // 8
	false,
	false,
	true,  // 11
	false,
	false,
	true,
	true,
	true,
	true
};

bool distKeypointComparator(
	const std::pair<double, hj::CKeyPoints> cmp1,
	const std::pair<double, hj::CKeyPoints> cmp2)
{
	return cmp1.first < cmp2.first;
}

hj::KeyPointsSet hj::RefineKeyPointTrajectories(hj::KeyPointsSet _inputSet)
{
	if (0 == _inputSet.size())
		return _inputSet;
	KeyPointsSet processedResult;

	// modify direction
	for (int kpIdx = 1; kpIdx < _inputSet.size(); ++kpIdx)
	{
		CKeyPoints wholeFlip = _inputSet[kpIdx].flip(0),
			upperFlip = _inputSet[kpIdx].flip(1), 
			lowerFlip = _inputSet[kpIdx].flip(2);
		
		double origDist = _inputSet[kpIdx-1].distance(_inputSet[kpIdx]),
			wholeFlipDist = _inputSet[kpIdx-1].distance(wholeFlip),
			upperFlipDist = _inputSet[kpIdx-1].distance(upperFlip),
			lowerFlipDist = _inputSet[kpIdx-1].distance(lowerFlip);

		std::vector<std::pair<double, CKeyPoints>> flipedKeyPoints;
		flipedKeyPoints.push_back(std::make_pair(origDist, _inputSet[kpIdx]));
		flipedKeyPoints.push_back(std::make_pair(wholeFlipDist, wholeFlip));
		flipedKeyPoints.push_back(std::make_pair(upperFlipDist, upperFlip));
		flipedKeyPoints.push_back(std::make_pair(lowerFlipDist, lowerFlip));

		// distance compare
		std::sort(flipedKeyPoints.begin(), flipedKeyPoints.end(), distKeypointComparator);		
		_inputSet[kpIdx] = flipedKeyPoints.front().second;
	}

	// fill time gap with dummy keypoints (when the gap is small enough)
	CKeyPoints dummyKeyPoints;
	dummyKeyPoints.fillZeroPoints();
	dummyKeyPoints.id = _inputSet[0].id;
	dummyKeyPoints.isSuspect = _inputSet[0].isSuspect;
	dummyKeyPoints.isThrowingGarbage = false;
	processedResult.push_back(_inputSet[0]);
	for (int kpIdx = 1; kpIdx < _inputSet.size(); ++kpIdx)
	{
		// fill with dummy keypoints
		if (kMaxTimeGap >= _inputSet[kpIdx].frameIndex - _inputSet[kpIdx - 1].frameIndex - 1)
		{
			dummyKeyPoints.isThrowingGarbage = _inputSet[kpIdx].isThrowingGarbage && _inputSet[kpIdx - 1].isThrowingGarbage;
			for (int i = _inputSet[kpIdx - 1].frameIndex + 1; i < _inputSet[kpIdx].frameIndex; ++i)
			{
				dummyKeyPoints.frameIndex = i;
				processedResult.push_back(dummyKeyPoints);
			}
		}

		// insert original keypoints
		processedResult.push_back(_inputSet[kpIdx]);
	}

	// interpolation
	for (int pIdx = 0; pIdx < NUM_KEYPOINT_TYPES; ++pIdx)
	{
		int procStart = -1;
		for (int kpIdx = 0; kpIdx < processedResult.size(); ++kpIdx)
		{
			if (kMinConfidence > processedResult[kpIdx].points[pIdx].confidence)
			{
				if (0 > procStart)
					procStart = kpIdx;
			}
			else
			{
				if (0 == procStart)
				{
					// boundary condition (cannot interpolate, because their is not lefthand side value
				}
				else if (0 < procStart && kMaxTimeGap >= processedResult[kpIdx].frameIndex - processedResult[procStart].frameIndex - 1)
				{
					// do interpolation
					hj::stKeyPoint prevPoint = processedResult[procStart-1].points[pIdx];
					hj::stKeyPoint currPoint = processedResult[kpIdx].points[pIdx];
					int prevFrameIdx = processedResult[procStart - 1].frameIndex;
					int currFrameIdx = processedResult[kpIdx].frameIndex;

					double denomInv = 1.0 / (double)(currFrameIdx - prevFrameIdx + 1);
					for (int procKpIdx = procStart; procKpIdx < kpIdx; ++procKpIdx)
					{
						double frontCoef = (double)(currFrameIdx - processedResult[procKpIdx].frameIndex);
						double backCoef = (double)(processedResult[procKpIdx].frameIndex - prevFrameIdx + 1);
						processedResult[procKpIdx].points[pIdx].x = denomInv * (frontCoef * prevPoint.x + backCoef * currPoint.x);
						processedResult[procKpIdx].points[pIdx].y = denomInv * (frontCoef * prevPoint.y + backCoef * currPoint.y);
						processedResult[procKpIdx].points[pIdx].confidence = kMinConfidence;
					}					
				}
				procStart = -1;  // end of processing window
			}
		}
	}

	// smoothing
	for (int pIdx = 0; pIdx < NUM_KEYPOINT_TYPES; ++pIdx)
	{
		if (!bSmoothing[pIdx])
			continue;

		std::vector<double> xs, ys;
		int startKpIdx = -1;
		for (int kpIdx = 0; kpIdx < processedResult.size(); ++kpIdx)
		{
			if (kMinConfidence > processedResult[kpIdx].points[pIdx].confidence // zero point
				|| (kpIdx > 0 && processedResult[kpIdx].frameIndex != processedResult[kpIdx - 1].frameIndex + 1)   // temporal discontinuity
				|| processedResult.size() - 1 == kpIdx)  // boundary condition
			{				
				if (0 > startKpIdx)
					continue;
				
				// do smoothing and replacing points with smoothed points
				CSGSmoother smootherX, smootherY;
				smootherX.Insert(xs);
				smootherY.Insert(ys);
				for (int i = 0, procKpIdx = startKpIdx; i < smootherX.size(); ++i, ++procKpIdx)
				{
					processedResult[procKpIdx].points[pIdx].x = smootherX.GetResult(i);
					processedResult[procKpIdx].points[pIdx].y = smootherY.GetResult(i);
				}
				xs.clear();
				ys.clear();

				startKpIdx = -1;
			}
			else
			{
				xs.push_back(processedResult[kpIdx].points[pIdx].x);
				ys.push_back(processedResult[kpIdx].points[pIdx].y);				
				if (0 > startKpIdx)
					startKpIdx = kpIdx;
			}
		}
	}


	return processedResult;
}



//()()
//('')HAANJU.YOO
