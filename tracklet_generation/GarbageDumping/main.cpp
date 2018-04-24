/**************************************************************************
* Title        : Online Multi-Camera Multi-Target Tracking Algorithm
* Author       : Haanju Yoo
* Initial Date : 2013.08.29 (ver. 0.9)
* Version Num. : 1.0 (since 2016.09.06)
* Description  :
*	The implementation of the paper named "Online Scheme for Multiple
*	Camera Multiple Target Tracking Based on Multiple Hypothesis 
*	Tracking" at IEEE transactions on Circuit and Systems for Video 
*	Technology (TCSVT).
***************************************************************************
                                            ....
                                           W$$$$$u
                                           $$$$F**+           .oW$$$eu
                                           ..ueeeWeeo..      e$$$$$$$$$
                                       .eW$$$$$$$$$$$$$$$b- d$$$$$$$$$$W
                           ,,,,,,,uee$$$$$$$$$$$$$$$$$$$$$ H$$$$$$$$$$$~
                        :eoC$$$$$$$$$$$C""?$$$$$$$$$$$$$$$ T$$$$$$$$$$"
                         $$$*$$$$$$$$$$$$$e "$$$$$$$$$$$$$$i$$$$$$$$F"
                         ?f"!?$$$$$$$$$$$$$$ud$$$$$$$$$$$$$$$$$$$$*Co
                         $   o$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                 !!!!m.*eeeW$$$$$$$$$$$f?$$$$$$$$$$$$$$$$$$$$$$$$$$$$$U
                 !!!!!! !$$$$$$$$$$$$$$  T$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                  *!!*.o$$$$$$$$$$$$$$$e,d$$$$$$$$$$$$$$$$$$$$$$$$$$$$$:
                 "eee$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$C
                b ?$$$$$$$$$$$$$$**$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!
                Tb "$$$$$$$$$$$$$$*uL"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
                 $$o."?$$$$$$$$F" u$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                  $$$$en '''    .e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
                   $$$B*  =*"?.e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$F
                    $$$W"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                     "$$$o#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    R: ?$$$W$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" :!i.
                     !!n.?$???""''.......,''''''"""""""""""''   ...+!!!
                      !* ,+::!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*'
                      "!?!!!!!!!!!!!!!!!!!!~ !!!!!!!!!!!!!!!!!!!~'
                      +!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!?!'
                    .!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!, !!!!
                   :!!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!!! '!!:
                .+!!!!!!!!!!!!!!!!!!!!!~~!! !!!!!!!!!!!!!!!!!! !!!.
               :!!!!!!!!!!!!!!!!!!!!!!!!!.':!!!!!!!!!!!!!!!!!:: '!!+
               "~!!!!!!!!!!!!!!!!!!!!!!!!!!.~!!!!!!!!!!!!!!!!!!!!.'!!:
                   ~~!!!!!!!!!!!!!!!!!!!!!!! ;!!!!~' ..eeeeeeo.'+!.!!!!.
                 :..    '+~!!!!!!!!!!!!!!!!! :!;'.e$$$$$$$$$$$$$u .
                 $$$$$$beeeu..  '''''~+~~~~~" ' !$$$$$$$$$$$$$$$$ $b
                 $$$$$$$$$$$$$$$$$$$$$UU$U$$$$$ ~$$$$$$$$$$$$$$$$ $$o
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$. $$$$$$$$$$$$$$$~ $$$u
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$$$ 8$$$$.
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$X $$$$$$$$$$$$$$'u$$$$$W
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$".$$$$$$$:
                 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$F.$$$$$$$$$
                 ?$$$$$$$$$$$$$$$$$$$$$$$$$$$$f $$$$$$$$$$$$' $$$$$$$$$$.
                  $$$$$$$$$$$$$$$$$$$$$$$$$$$$ $$$$$$$$$$$$$  $$$$$$$$$$!
                  "$$$$$$$$$$$$$$$$$$$$$$$$$$$ ?$$$$$$$$$$$$  $$$$$$$$$$!
                   "$$$$$$$$$$$$$$$$$$$$$$$$Fib ?$$$$$$$$$$$b ?$$$$$$$$$
                     "$$$$$$$$$$$$$$$$$$$$"o$$$b."$$$$$$$$$$$  $$$$$$$$'
                    e. ?$$$$$$$$$$$$$$$$$ d$$$$$$o."?$$$$$$$$H $$$$$$$'
                   $$$W.'?$$$$$$$$$$$$$$$ $$$$$$$$$e. "??$$$f .$$$$$$'
                  d$$$$$$o "?$$$$$$$$$$$$ $$$$$$$$$$$$$eeeeee$$$$$$$"
                  $$$$$$$$$bu "?$$$$$$$$$ 3$$$$$$$$$$$$$$$$$$$$*$$"
                 d$$$$$$$$$$$$$e. "?$$$$$:'$$$$$$$$$$$$$$$$$$$$8
         e$$e.   $$$$$$$$$$$$$$$$$$+  "??f "$$$$$$$$$$$$$$$$$$$$c
        $$$$$$$o $$$$$$$$$$$$$$$F"          '$$$$$$$$$$$$$$$$$$$$b.0
       M$$$$$$$$U$$$$$$$$$$$$$F"              ?$$$$$$$$$$$$$$$$$$$$$u
       ?$$$$$$$$$$$$$$$$$$$$F                   "?$$$$$$$$$$$$$$$$$$$$u
        "$$$$$$$$$$$$$$$$$$"                       ?$$$$$$$$$$$$$$$$$$$$o
          "?$$$$$$$$$$$$$F                            "?$$$$$$$$$$$$$$$$$$
             "??$$$$$$$F                                 ""?3$$$$$$$$$$$$F
                                                       .e$$$$$$$$$$$$$$$$'
                                                      u$$$$$$$$$$$$$$$$$
                                                     '$$$$$$$$$$$$$$$$"
                                                      "$$$$$$$$$$$$F"
                                                        ""?????""

**************************************************************************/

#include <sstream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "MTTracker.h"


#define KEYPOINTS_BASE_PATH ("D:\Workspace\Dataset\ETRI\GarbageDumping\pose_text")
#define VIDEO_BASE_PATH ("D:\\Workspace\\Dataset\\ETRI\\GarbageDumping\\numbering")
#define RESULT_PATH ("D:\\Workspace\\ExperimentalResult\\ETRI\\GarbageDumpingResult")
#define HEATMAP_PATH ("../output_heatmaps")
#define TRAINED_MODEL_PATH ("../model")

#define TARGET_VIDEO ("172") // for staying: 165. start from 165.
#define START_FRAME_INDEX (0)
#define END_FRAME_INDEX (-1)


int main(int argc, char** argv)
{	
	// read video info
	std::string strVideoPath = std::string(VIDEO_BASE_PATH) + "\\" + std::string(TARGET_VIDEO) + ".mp4";
	cv::VideoCapture *pVideoCapture = new cv::VideoCapture(strVideoPath);
	int nLastFrameIndex = END_FRAME_INDEX < 0 ? 
		(int)pVideoCapture->get(CV_CAP_PROP_FRAME_COUNT) : 
		std::min((int)END_FRAME_INDEX, (int)pVideoCapture->get(CV_CAP_PROP_FRAME_COUNT));
	int imageWidth = (int)pVideoCapture->get(CV_CAP_PROP_FRAME_WIDTH), 
		imageHeight = (int)pVideoCapture->get(CV_CAP_PROP_FRAME_HEIGHT);


	//---------------------------------------------------
	// TRACKER INITIATION
	//---------------------------------------------------
	hj::CTrackResult trackResult;     // <- The tracking result will be saved here
	hj::stParamTrack trackParams;     // <- Contains whole parameters of tracking module. Using default values is recommended.
	trackParams.nImageWidth = imageWidth;
	trackParams.nImageHeight = imageHeight;
	trackParams.dImageRescale = 1.0;  // <- Heavy influence on the speed of the algorithm.
//	trackParams.bVisualize = true;
//	trackParams.bVideoRecord = true;  // <- To recoder the result visualization.
	trackParams.bVisualize = true;
	trackParams.bVideoRecord = false;  // <- To recoder the result visualization.

	trackParams.strVideoRecordPath = std::string(RESULT_PATH) + "\\" + std::string(TARGET_VIDEO);
	hj::CMTTracker cTracker;      // <- The instance of a multi-target tracker.
	cTracker.Initialize(trackParams);
		

	//---------------------------------------------------
	// MAIN LOOP FOR TRACKING
	//---------------------------------------------------
	std::string strKeypointsBasePath = std::string(KEYPOINTS_BASE_PATH) + "\\" + std::string(TARGET_VIDEO);
	std::string strFilePath;  // <- temporary file path for this and that
	cv::Mat matCurFrame;
	cv::Mat matCurFrame_re; // resized image for background modeling
	hj::KeyPointsSet curKeyPoints;

	// Pose Heatmap related.
	std::string strHeatPath = std::string(HEATMAP_PATH);
	std::string strFilePath_heat;

	// DISPLAY
	//cv::Mat matDisp;
	//std::string strWindowName = "Result";

	// display control
	bool bPaused = false;

	for (int fIdx = START_FRAME_INDEX; fIdx < nLastFrameIndex; fIdx++)
	{
		// Grab frame image
		(*pVideoCapture) >> matCurFrame;
		//matDisp = matCurFrame.clone();

		// Read keypoints
		strFilePath = strKeypointsBasePath + "\\"
			+ std::string(TARGET_VIDEO) + hj::FormattedString("_%012d_keypoints.txt", fIdx);
		curKeyPoints = hj::ReadKeypoints(strFilePath);

		// Track targets between consecutive frames
		trackResult = cTracker.Track(curKeyPoints, matCurFrame, fIdx);

		// Load the Pose Background Heatmap
		strFilePath_heat = strHeatPath + "\\"
			+ std::string(TARGET_VIDEO) + hj::FormattedString("_%012d_heatmaps.png", fIdx);
		cv::Mat Heatmap = cv::imread(strFilePath_heat);
		cv::cvtColor(Heatmap, Heatmap, CV_BGR2GRAY);

		// Visualization	
		//cv::imshow(strWindowName, matDisp);
		//cvWaitKey(0.1);
		//matDisp.release();
		do {
			int pressedKey = cv::waitKey(30);
			if ((int)' ' == pressedKey) {
				bPaused = !bPaused;
				if (!bPaused)
					break;
			}
			else if (bPaused && (int)'f' == pressedKey) { break; }
		} while (bPaused);
	}

	//cv::destroyWindow(strWindowName);
	
	
	return 0;
}



//()()
//('')HAANJU.YOO
