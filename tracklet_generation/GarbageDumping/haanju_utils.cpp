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


hj::KeyPointsSet hj::ReadKeypoints(const std::string _strFilePath)
{
	KeyPointsSet vec_keypoints;
	int num_people = 0;
	double x = 0.0, y = 0.0, confidence = 0.0;

	FILE *fp = NULL;
	try {
		fopen_s(&fp, _strFilePath.c_str(), "r");
		if (NULL == fp) { return vec_keypoints; }

		fscanf_s(fp, "%d\n", &num_people);
		vec_keypoints.resize(num_people);

		// read box infos
		for (int row = 0; row < num_people; row++)
		{	
			double minX = DBL_MAX, maxX = 0.0, minY = DBL_MAX, maxY = 0.0;
			vec_keypoints[row].points.resize(NUM_KEYPOINT_TYPES);
			for (int pIdx = 0; pIdx < NUM_KEYPOINT_TYPES; pIdx++)
			{
				fscanf_s(fp, "%lf,%lf,%lf,", &x, &y, &confidence);
				vec_keypoints[row].points[pIdx].x = x;
				vec_keypoints[row].points[pIdx].y = y;
				vec_keypoints[row].points[pIdx].confidence = confidence;

				// skip invalid points
				if (0.0 == x && 0.0 == y) { continue; }

				if (minX > x) { minX = x; } if (maxX < x) { maxX = x; }
				if (minY > y) { minY = y; }	if (maxY < y) { maxY = y; }
			}
			fscanf_s(fp, "\n");
			vec_keypoints[row].bbox = cv::Rect2d(minX, minY, maxX - minX + 1.0, maxY - minY + 1.0);
		}
		fclose(fp);
	}
	catch (int nError) {
		printf("[ERROR] file open error with detection result reading: %d\n", nError);
	}
	return vec_keypoints;
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



//()()
//('')HAANJU.YOO
