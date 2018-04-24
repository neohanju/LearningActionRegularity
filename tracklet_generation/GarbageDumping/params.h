#ifndef	_PARAMS_H_
#define	_PARAMS_H_



#define BLOCK_SIZE_X					(2.0)
#define BLOCK_SIZE_Y					(2.0)
#define BLOCK_SIZE_SQR				(BLOCK_SIZE_X*BLOCK_SIZE_Y)
#define VARIANCE_INTERPOLATE_PARAM	(1.0)


#define MAX_BG_AGE				(100.0)
#define VAR_MIN_NOISE_T			(50.0*50.0)
#define VAR_DEC_RATIO			(0.001)
#define MIN_BG_VAR				(5.0*5.0)	//15*15
#define INIT_BG_VAR				(20.0*20.0)	//15*15


#define NUM_MODELS		(2)
#define VAR_THRESH_FG_DETERMINE		(4.0)
#define VAR_THRESH_MODEL_MATCH		(2.0)





#endif // _PARAMS_H_
 