#pragma once

//if you do not use opencv test, please comment out _OPENCV_ define.
#define _OPENCV_ 


/*
<mmintrin.h>  MMX
<xmmintrin.h> SSE
<emmintrin.h> SSE2
<pmmintrin.h> SSE3
<tmmintrin.h> SSSE3
<smmintrin.h> SSE4.1
<nmmintrin.h> SSE4.2
<ammintrin.h> SSE4A
<wmmintrin.h> AES
<immintrin.h> AVX
*/
#include <nmmintrin.h>
#include <iostream>
#include <vector>
#include <string>
using namespace std;
typedef unsigned char uchar ;

//timer for windows
#include <windows.h>
class timer
{
	string message;
	LARGE_INTEGER freq;
	LARGE_INTEGER begin;
	LARGE_INTEGER end;
public:
	timer()
	{
		message="";
		;
	}
	timer(string message_)
	{
		message=message_;
		;
	}
	void start()
	{
		QueryPerformanceFrequency(&freq );
		QueryPerformanceCounter(&begin );
	}
	void stop()
	{
		QueryPerformanceCounter(&end );

		printf( "%f s\n", ( double )( end.QuadPart - begin.QuadPart ) / freq.QuadPart );
	}
};

#ifdef _OPENCV_
#include <opencv2/opencv.hpp>
#include <opencv2/core/internal.hpp>
using namespace cv;
using namespace std;

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#endif
#endif