#include "SimdProgrammingLesson.h"

bool checkData(float* src1, float* src2, int size)
{
	bool ret= true;
	for(int i=0;i<size;i++)
	{
		if(src1[i]!=src2[i])
		{
			ret=false;
			break;
		}
	}
	return ret;
}

void showResult(bool flag, string message)
{
	if(flag==true)
		cout<<message<<" OK"<<endl;
	else
		cout<<message<<" NG"<<endl;
}

void printData(float* data, int size)
{
	for(int i=0;i<size;i++)
	{
		cout<<data[i]<<endl;
	}
}


float* createAlign16Data_32f(int size)
{
	float* ret = (float*)_mm_malloc(sizeof(float)*size,  16);
	return ret;
}

void setLinearData_32f(float* data, int size)
{
	for(int i=0;i<size;i++)
	{
		data[i]=(float)i;
	}
}

void releaseData(float* data)
{
	_mm_free(data);
}