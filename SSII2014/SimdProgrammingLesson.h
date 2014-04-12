#pragma once 

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
#include <string>
using namespace std;

//for data alloc
void releaseData(float* data);
void setLinearData_32f(float* data, int size);
float* createAlign16Data_32f(int size);

//for utility functions
void printData(float* data, int size);
bool checkData(float* src1, float* src2, int size);
void showResult(bool flag, string message);
