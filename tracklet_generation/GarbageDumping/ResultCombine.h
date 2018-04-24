#pragma once
#include <opencv2/highgui/highgui.hpp>

struct stParamResult
{
	stParamResult()
		: bVisualize(true)
	{};

	~stParamResult() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------

	bool bVisualize;
};

class ResultCombine
{
	//----------------------------------------------------------------
	// METHODS
	//---------------------------------------------------------------
public:
	ResultCombine();
	~ResultCombine();

	void Initialize();
	void Finalize();

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
public:
	bool bInit_;

	/* visualization related */
	bool             bVisualizeResult_;
	std::string      strVisWindowName_;

	/*Detection relate*/
	bool bSVMResult;
	bool bThrowResult;

};


