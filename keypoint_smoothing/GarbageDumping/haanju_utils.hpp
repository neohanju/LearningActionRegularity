#pragma once

#include <sstream>
#include <memory>    // For std::unique_ptr
#include <stdarg.h>  // For va_start, etc.
#include "haanju_types.hpp"

namespace hj
{

//=============================================================================
// POINT OPERATIONS
//=============================================================================

/************************************************************************
 Method Name: NormL2
 Description:
	- Get the L2-norm value of cv::Point.
 Input Arguments:
	- _point: target point.
 Return Values:
	- L2-norm value of the input point.
************************************************************************/
template<typename _Tp>
double NormL2(cv::Point_<_Tp> _point)
{
	return sqrt((double)(_point.x * _point.x + _point.y * _point.y));
}



//=============================================================================
// RECT OPERATIONS
//=============================================================================

/************************************************************************
 Method Name: NormL2
 Description:
	- Get the L2-norm value of a cv::Point.
 Input Arguments:
	- _point: Target point.
 Return Values:
	- double: The L2-norm value of the input point.
************************************************************************/
template<typename _Tp>
cv::Rect_<_Tp> Rescale(const cv::Rect_<_Tp> &_target_rect, double _scale)
{
	cv::Rect_<_Tp> rescaled_rect = _target_rect;
	rescaled_rect.x = (_Tp)_scale * rescaled_rect.x;
	rescaled_rect.y = (_Tp)_scale * rescaled_rect.y;
	rescaled_rect.width = (_Tp)_scale * rescaled_rect.width;
	rescaled_rect.height = (_Tp)_scale * rescaled_rect.height;
	return rescaled_rect;
}


/************************************************************************
 Method Name: Center
 Description:
	- Get the center point of a cv::Rect.
 Input Arguments:
	- _rect: Target rect.
 Return Values:
	- cv::Point: The center point of the input rect.
************************************************************************/
template<typename _Tp>
cv::Point_<_Tp> Center(const cv::Rect_<_Tp> _rect)
{
	return cv::Point_<_Tp>(_rect.x + (_Tp)(0.5 * _rect.width), _rect.y + (_Tp)(0.5 * _rect.height));
}

/************************************************************************
 Method Name: CropWithSize
 Description:
	- Crop the input box with the maximum ranges of each witdh and height.
 Input Arguments:
	- _rect: Target rect.
	- _size: The maximum values of witdh and height.
 Return Values:
	- cv::Rect: Cropped rect.
************************************************************************/
template<typename _Tp>
cv::Rect_<_Tp>  CropWithSize(const cv::Rect_<_Tp> &_rect, const cv::Size _size)
{
	cv::Rect_<_Tp> croppedRect = _rect;
	croppedRect.x = std::max((_Tp)0.0, _rect.x);
	croppedRect.y = std::max((_Tp)0.0, _rect.y);
	croppedRect.width  = std::min(_rect.width, (_Tp)_size.width - croppedRect.x - 1);
	croppedRect.height = std::min(_rect.height, (_Tp)_size.height - croppedRect.y - 1);
	return croppedRect;
}

/************************************************************************
 Method Name: CheckOverlap
 Description:
	- Check whether two rects overlap each other.
 Input Arguments:
	- _rect1: The 1st rect.
	- _rect2: The 2nd rect.
 Return Values:
	- bool: (true) overlap each other / (false) do not overlap.
************************************************************************/
template<typename _Tp>
bool CheckOverlap(const cv::Rect_<_Tp> _rect1, const cv::Rect_<_Tp> _rect2)
{
	if (std::max(_rect1.x + _rect1.width, _rect2.x + _rect2.width) - std::min(_rect1.x, _rect2.x) >= _rect1.width + _rect2.width)
		return false;
	if (std::max(_rect1.y + _rect1.height, _rect2.y + _rect2.height) - std::min(_rect1.y, _rect2.y) >= _rect1.height + _rect2.height)
		return false;
	return true;
}

/************************************************************************
 Method Name: OverlappedArea
 Description:
	- Get the area of the overlapping region of two rects.
 Input Arguments:
	- _rect1: The 1st rect.
	- _rect2: The 2nd rect.
 Return Values:
	- double: The area of the overlapping region of the boxes.
************************************************************************/
template<typename _Tp>
double OverlappedArea(const cv::Rect_<_Tp> &_rect1, const cv::Rect_<_Tp> &_rect2)
{
	double overlappedWidth = std::min(_rect1.x + _rect1.width, _rect2.x + _rect2.width) - std::max(_rect1.x, _rect2.x);
	if (0.0 >= overlappedWidth)
		return 0.0;
	double overlappedHeight = std::min(_rect1.y + _rect1.height, _rect2.y + _rect2.height) - std::max(_rect1.y, _rect2.y);
	if (0.0 >= overlappedHeight)
		return 0.0;
	return overlappedWidth * overlappedHeight;
}


//=============================================================================
// STRING MANUPULATION
//=============================================================================

std::string FormattedString(const std::string _formatted_string, ...);


//=============================================================================
// ETC
//=============================================================================


////////////////////////////////////////////////////////////////////////
// IMAGE BUFFER
/////////////////////////////////////////////////////////////////////////
class CMatFIFOBuffer
{
	//------------------------------------------------
	// METHODS
	//------------------------------------------------
public:
	CMatFIFOBuffer() : bInit_(false) {}
	CMatFIFOBuffer(int _bufferSize) : bInit_(true), bufferSize_(_bufferSize) {}
	~CMatFIFOBuffer() { clear(); }
	bool    set(int _bufferSize);
	bool    clear();
	bool    insert(cv::Mat _newMat);
	bool    insert_resize(cv::Mat _newMat, cv::Size _resizeParam);
	bool    remove(int _pos);
	size_t  size() { return bufferSize_; }
	size_t  num_elements() { return buffer_.size(); }
	cv::Mat front() { assert(bInit_); return buffer_.front(); }
	cv::Mat back() { assert(bInit_); return buffer_.back(); }
	cv::Mat get(int _pos) { assert(bInit_ && _pos < bufferSize_); return buffer_[_pos]; }
	int     get_back_idx() { return (int)buffer_.size(); }

	/* iterators */
	typedef std::deque<cv::Mat>::iterator iterator;
	typedef std::deque<cv::Mat>::const_iterator const_iterator;
	typedef std::deque<cv::Mat>::reverse_iterator reverse_iterator;
	typedef std::deque<cv::Mat>::const_reverse_iterator const_reverse_iterator;
	iterator begin() { return buffer_.begin(); }
	iterator end() { return buffer_.end(); }
	const_iterator begin() const { return buffer_.begin(); }
	const_iterator end()   const { return buffer_.end(); }
	reverse_iterator rbegin() { return buffer_.rbegin(); }
	reverse_iterator rend() { return buffer_.rend(); }
	const_reverse_iterator rbegin() const { return buffer_.rbegin(); }
	const_reverse_iterator rend()   const { return buffer_.rend(); }


	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
private:
	bool bInit_;
	int  bufferSize_;
	std::deque<cv::Mat> buffer_;
};


/************************************************************************
 Method Name: ReadKeypoints
 Description:
	- Read keypoints from pre-generated txt files.
 Input Arguments:
	- _strFilePath : Path to an input file
 Return Values:
	- KeyPointsSet: Read keypoints.
************************************************************************/
std::vector<KeyPointsSet> ReadKeypoints(const std::string _file_path);
bool WriteKeypoints(const std::string _file_path, const std::vector<KeyPointsSet> _vec_key_point_sets);


double KeyPointsDistance(
	CKeyPoints _firstKeypoints,
	CKeyPoints _secondKeypoints,
	bool _average = false,
	std::vector<double> *_ptPointWeights = NULL);

bool CheckOverlap(const CKeyPoints _firstKeyPoints, const CKeyPoints _secondKeyPoints);

KeyPointsSet RefineKeyPointTrajectories(const KeyPointsSet _inputSet);

}

//()()
//('')HAANJU.YOO
