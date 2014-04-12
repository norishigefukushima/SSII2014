#include "SimdProgrammingLesson.h"

void lesson_1_01_test(float* src, float* dest, int size)
{
	for(int i=0;i<size;i++)
	{
		dest[i]=src[i];
	}
}

void lesson_1_01_answer(float* src, float* dest, int size)
{
	int ssesize = size/4;
	for(int i=0;i<ssesize;i++)
	{
		__m128 ms = _mm_load_ps(src);
		_mm_store_ps(dest, ms);
		src+=4, dest+=4;
	}
}


//hint
//_mm_load_ps
//_mm_store_ps
void lesson_1_01(float* src, float* dest, int size)
{
	;
}


bool main_lesson_1_01()
{
	bool ret=false;
	const int size = 32;
	float* src1 = createAlign16Data_32f(size);
	float* dst1 = createAlign16Data_32f(size);
	float* dstTest = createAlign16Data_32f(size);

	setLinearData_32f(src1, size);

	lesson_1_01_test(src1,dstTest, size);
	lesson_1_01(src1, dst1, size);
	bool result01 = checkData(dst1, dstTest, size);
	showResult(result01, "Lesson 1-01");


	releaseData(src1);
	releaseData(dst1);
	releaseData(dstTest);
	return 0;
}

//float
//copy, alloc, set
//basic arithmetic: map
//advance arithmetic: map
//comparison
//reduction
//shuffle

// u/char, u/short, u/int
//copy, alloc, set
//convert
//basic arithmetic: map
//advance arithmetic: map
//comparison
//reduction
//shuffle

//application
//image processing?
int main()
{
	main_lesson_1_01();
	return 0;
}