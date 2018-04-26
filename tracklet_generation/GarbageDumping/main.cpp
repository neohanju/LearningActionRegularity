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
#include <time.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "MTTracker.h"


#define KEYPOINTS_BASE_PATH ("D:\\Workspace\\Dataset\\ETRI\\GarbageDumping\\pose_text")
#define VIDEO_BASE_PATH ("D:\\Workspace\\Dataset\\ETRI\\GarbageDumping\\numbering")
#define RESULT_PATH ("D:\\Workspace\\ExperimentalResult\\ETRI\\GarbageDumpingResult")
//#define HEATMAP_PATH ("../output_heatmaps")
//#define TRAINED_MODEL_PATH ("../model")

#define TARGET_VIDEO ("061")
#define START_FRAME_INDEX (0)
#define END_FRAME_INDEX (-1)

//#define VIDEO_064

#ifdef VIDEO_006  // keypoints are not extracted well
	#define TARGET_VIDEO ("006") // for staying: 165. start from 165.
	#define START_FRAME_INDEX (120)
	#define END_FRAME_INDEX (420)
#endif
#ifdef VIDEO_010  // keypoints are not extracted well
	#define TARGET_VIDEO ("010") // for staying: 165. start from 165.
	#define START_FRAME_INDEX (50)
	#define END_FRAME_INDEX (-1)
#endif
#ifdef VIDEO_064
	#define TARGET_VIDEO ("064") // for staying: 165. start from 165.
	#define START_FRAME_INDEX (0)
	#define END_FRAME_INDEX (-1)
#endif
#ifdef VIDEO_172
	#define TARGET_VIDEO ("172") // for staying: 165. start from 165.
	#define START_FRAME_INDEX (270)
	#define END_FRAME_INDEX (500)
#endif


int main(int argc, char** argv)
{	
	for (int vIdx = 52; vIdx <= 219; ++vIdx) {		
		char strVideoNameChar[128];
		sprintf_s(strVideoNameChar, "%03d", vIdx);
		std::string strVideoName(strVideoNameChar);
		//std::string strVideoName = std::string(TARGET_VIDEO);		

		// read video info
		std::string strVideoPath = std::string(VIDEO_BASE_PATH) + "\\" + strVideoName + ".mp4";
		cv::VideoCapture *pVideoCapture = new cv::VideoCapture(strVideoPath);
		int nLastFrameIndex = END_FRAME_INDEX < 0 ? 
			(int)pVideoCapture->get(CV_CAP_PROP_FRAME_COUNT) : 
			std::min((int)END_FRAME_INDEX, (int)pVideoCapture->get(CV_CAP_PROP_FRAME_COUNT));
		int imageWidth = (int)pVideoCapture->get(CV_CAP_PROP_FRAME_WIDTH), 
			imageHeight = (int)pVideoCapture->get(CV_CAP_PROP_FRAME_HEIGHT);

		// set starting frame
		pVideoCapture->set(CV_CAP_PROP_POS_FRAMES, std::min((int)START_FRAME_INDEX, nLastFrameIndex));


		printf("%s: %d frames\n", strVideoName.c_str(), nLastFrameIndex);

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

		trackParams.strVideoRecordPath = std::string(RESULT_PATH) + "\\" + strVideoName;
		hj::CMTTracker cTracker;      // <- The instance of a multi-target tracker.
		cTracker.Initialize(trackParams);
		

		//---------------------------------------------------
		// MAIN LOOP FOR TRACKING
		//---------------------------------------------------
		std::string strFilePath;  // <- temporary file path for this and that
		cv::Mat matCurFrame;	
		hj::KeyPointsSet curKeyPoints;
	
		// display control
		bool bPaused = true;
		time_t elapsedTime;

		for (int fIdx = START_FRAME_INDEX; fIdx < nLastFrameIndex; fIdx++)
		{
			elapsedTime = time(NULL);

			// Grab frame image
			(*pVideoCapture) >> matCurFrame;

			// Read keypoints
			strFilePath = std::string(KEYPOINTS_BASE_PATH) + "\\" + strVideoName + "\\"
				+ hj::FormattedString("%s_%012d_keypoints.txt", strVideoName.c_str(), fIdx);
			curKeyPoints = hj::ReadKeypoints(strFilePath);

			// Track targets between consecutive frames
			trackResult = cTracker.Track(curKeyPoints, matCurFrame, fIdx);

			// display control
			do {
				long long waitingTime = std::max((long long)5, 30 + elapsedTime - time(NULL));
				int pressedKey = cv::waitKey((int)waitingTime);
				if ((int)' ' == pressedKey) {
					bPaused = !bPaused;
					if (!bPaused)
						break;
				}
				else if (bPaused && (int)'f' == pressedKey) { break; }
			} while (bPaused);
		}
	}	
	
	return 0;
}



//()()
//('')HAANJU.YOO
