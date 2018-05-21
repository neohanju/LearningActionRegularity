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
#include "haanju_utils.hpp"
#include "dirent.h"


#define KEYPOINTS_BASE_PATH ("keypoints")
#define RESULT_PATH ("keypoints\\processed")
const std::string kExtension("txt");

int main(int argc, char** argv)
{	
	DIR *dir;
	struct dirent *ent;

	std::vector<std::string> vecTrajectoryFileNames;
	if ((dir = opendir(KEYPOINTS_BASE_PATH)) != NULL) 
	{
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) 
		{
			if (ent->d_type != DT_REG)
				continue;
			// if entry is a regular file
			std::string fname = ent->d_name;	// filename												
			if (fname.find(kExtension, (fname.length() - kExtension.length())) != std::string::npos)
				vecTrajectoryFileNames.push_back(fname);		// add filename to results vector
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}



	for (int vIdx = 1; vIdx <= 219; ++vIdx)
	//for (int vIdx = 41; vIdx <= 41; ++vIdx)
	{
		char strKeypointsFileName[128];
		sprintf_s(strKeypointsFileName, "%06d.txt", vIdx);		
				
		// read keypoints
		std::string strKeypointsFilePath = std::string(KEYPOINTS_BASE_PATH) + "\\" + std::string(strKeypointsFileName);
		std::vector<hj::KeyPointsSet> vecKeyPointsSet = hj::ReadKeypoints(strKeypointsFilePath);
		if (0 == vecKeyPointsSet.size())
			continue;

		// refine keypoints
		printf("processing %s\n", strKeypointsFileName);
		std::vector<hj::KeyPointsSet> vecProcessedSet;
		for (int personIdx = 0; personIdx < vecKeyPointsSet.size(); ++personIdx)
		{
			vecProcessedSet.push_back(hj::RefineKeyPointTrajectories(vecKeyPointsSet[personIdx]));
		}

		// write result keypoints
		std::string strResultFilePath = std::string(RESULT_PATH) + "\\" + std::string(strKeypointsFileName);
		hj::WriteKeypoints(strResultFilePath, vecProcessedSet);
	}	
	
	return 0;
}



//()()
//('')HAANJU.YOO
