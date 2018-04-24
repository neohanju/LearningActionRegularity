#include "ResultCombine.h"
#include <opencv2/imgproc/imgproc.hpp>



ResultCombine::ResultCombine()
	:bInit_(false)
{
}


ResultCombine::~ResultCombine()
{
	Finalize();
}

void ResultCombine::Initialize()
{
	if (bInit_) { Finalize(); }

	//stParam_ = _stParam;
	bInit_ = true;

	// visualization related
	//bVisualizeResult_ = stParam_.bVisualize;
	strVisWindowName_ = "Final result";

}

void ResultCombine::Finalize()
{
	if (!bInit_) { return; }

	bInit_ = false;

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }
}
