#include "SimdProgrammingLesson.h"

typedef unsigned char uchar ;

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
	//main_lesson_1_01();
	return 0;
}



void add(uchar* a, uchar* b, uchar* dest, int num)
{
	for(int i=0;i<num;i++)
	{
		dest[i] = a[i] + b[i];
	}
}

void addOpenMP(uchar* a, uchar* b, uchar* dest, int num)
{
#pragma omp parallel for
	for(int i=0;i<num;i++)
	{
		dest[i] = a[i] + b[i];
	}
}

void add_sse_uchar(uchar* a, uchar* b, uchar* dest, int num)
{
	for(int i=0;i<num;i+=16)
	{
		//メモリ上の配列A，Bを各をレジスタへロード
		__m128i ma = _mm_load_si128((const __m128i*)(a+i));
		__m128i mb = _mm_load_si128((const __m128i*)(b+i));
		//A,Bが保持されたレジスタの内容を加算してmaのレジスタにコピー
		ma = _mm_add_epi8(ma,mb);
		//計算結果のレジスタ内容をメモリ（dest）にストア
		_mm_store_si128((__m128i*)(dest+i), ma);

	}
}


void add_sse_float(float* a, float* b, float* dest, int num)
{
	for(int i=0;i<num;i+=4)
	{
		//メモリ上の配列A，Bを各をレジスタへロード
		__m128 ma = _mm_load_ps((a+i));
		__m128 mb = _mm_load_ps((b+i));
		//A,Bが保持されたレジスタの内容を加算してmaのレジスタにコピー
		ma = _mm_add_ps(ma,mb);
		//計算結果のレジスタ内容をメモリ（dest）にストア
		_mm_store_ps((dest+i), ma);
	}
}

void add_avx_float(float* a, float* b, float* dest, int num)
{
	for(int i=0;i<num;i+=8)
	{
		//メモリ上の配列A，Bを各をレジスタへロード
		__m256 ma = _mm256_load_ps((a+i));
		__m256 mb = _mm256_load_ps((b+i));
		//A,Bが保持されたレジスタの内容を加算してmaのレジスタにコピー
		
		ma = _mm256_add_ps(ma,mb);
		//計算結果のレジスタ内容をメモリ（dest）にストア
		_mm256_store_ps((dest+i), ma);
	}
}

float sum(float* src, int num)
{
	float ret=0.0f;
	for(int i=0;i<num;i++)
	{
		ret += src[i];
	}
	return ret;
}

float sum2(float* src, int num)
{
	float ret0=0.0f;
	float ret1=0.0f;
	float ret2=0.0f;
	float ret3=0.0f;

	for(int i=0;i<num;i+=4)
	{
		ret0 += src[4*i+0];
		ret1 += src[4*i+1];
		ret2 += src[4*i+2];
		ret3 += src[4*i+3];
	}

	return ret0+ret1+ret2+ret3;
}

float sumOMP(float* src, int num)
{
	float ret=0.0f;
#pragma omp parallel for
	for(int i=0;i<num;i++)
	{
		ret += src[i];
	}
	return ret;
}

float sumOMPTrue(float* src, int num)
{
	float ret=0.0f;
#pragma omp parallel for (+:ret)
	for(int i=0;i<num;i++)
	{
		ret += src[i];
	}
	return ret;
}

void iirfilter(float* src, float* dest, int w, int h, float a)
{
	float ia = 1.0f-a;
	for(int j=1;j<h-1;j++)//画像端を無視
	{
		for(int i=1;i<w-1;i++)
		{
			float sum = 0.0f;
			sum+= src[w*(j+l) + i+l];
			
			dest[w*j+i] = sum*normalize;
		}
	}
}

void boxfilter(float* src, float* dest, int w, int h, int r)
{
	float	normalize = 1.0f/(float)((2*r+1)*(2*r+1));
	for(int j=r;j<h-r;j++)//画像端を無視
	{
		for(int i=r;i<w-r;i++)
		{
			float sum = 0.0f;
			for(int l=-r;l<=r;l++)
			{
				for(int k=-r;k<=r;k++)
				{
					sum+= src[w*(j+l) + i+l];
				}
			}
			dest[w*j+i] = sum*normalize;
		}
	}
}


void boxfilter_omp1(float* src, float* dest, int w, int h, int r)
{
	float	normalize = 1.0f/(float)((2*r+1)*(2*r+1));
#pragma omp parallel for
	for(int j=r;j<h-r;j++)//画像端を無視
	{
		for(int i=r;i<w-r;i++)
		{
			float sum = 0.0f;
			for(int l=-r;l<=r;l++)
			{
				for(int k=-r;k<=r;k++)
				{
					sum+= src[w*(j+l) + i+l];
				}
			}
			dest[w*j+i] = sum*normalize;
		}
	}
}

void boxfilter_omp2(float* src, float* dest, int w, int h, int r)
{
	float	normalize = 1.0f/(float)((2*r+1)*(2*r+1));
	for(int j=r;j<h-r;j++)//画像端を無視
	{
		#pragma omp parallel for
		for(int i=r;i<w-r;i++)
		{
			float sum = 0.0f;
			for(int l=-r;l<=r;l++)
			{
				for(int k=-r;k<=r;k++)
				{
					sum+= src[w*(j+l) + i+l];
				}
			}
			dest[w*j+i] = sum*normalize;
		}
	}
}

void boxfilter_omp3(float* src, float* dest, int w, int h, int r)
{
	float	normalize = 1.0f/(float)((2*r+1)*(2*r+1));

	for(int j=r;j<h-r;j++)//画像端を無視
	{
		for(int i=r;i<w-r;i++)
		{
			float sum = 0.0f;
			#pragma omp parallel for reduction(+:sum)
			for(int l=-r;l<=r;l++)
			{
				for(int k=-r;k<=r;k++)
				{
					sum+= src[w*(j+l) + i+l];
				}
			}
			dest[w*j+i] = sum*normalize;
		}
	}
}



void boxfilter_sse(float* src, float* dest, int w, int h, int r)
{
	float	normalize = 1.0f/(float)((2*r+1)*(2*r+1));
	__m128 mnormalize = _mm_set1_ps(normalize);
	for(int j=r;j<h-r;j++)//画像端を無視
	{
		for(int i=r;i<w-r;i+=4)
		{
			__m128 msum = _mm_setzero_ps();
			for(int l=-r;l<=r;l++)
			{
				for(int k=-r;k<=r;k++)
				{
					__m128 ms = _mm_loadu_ps((src + w*(j+l) + i+l));
					msum = _mm_add_ps(msum,ms);
				}
			}
			msum = _mm_mul_ps(msum,mnormalize);
			_mm_storeu_ps(dest+w*j+i,msum);
		}
	}
}

void boxfilter_sse_omp(float* src, float* dest, int w, int h, int r)
{

#pragma omp parallel for
	for(int j=r;j<h-r;j++)//画像端を無視
	{
		float	normalize = 1.0f/(float)((2*r+1)*(2*r+1));
		__m128 mnormalize = _mm_set1_ps(normalize);
		for(int i=r;i<w-r;i+=4)
		{
			__m128 msum = _mm_setzero_ps();
			for(int l=-r;l<=r;l++)
			{
				for(int k=-r;k<=r;k++)
				{
					__m128 ms = _mm_loadu_ps((src + w*(j+l) + i+l));
					msum = _mm_add_ps(msum,ms);
				}
			}
			msum = _mm_mul_ps(msum,mnormalize);
			_mm_storeu_ps(dest+w*j+i,msum);
		}
	}
}

float sum_sse_float(float* src, int num)
{
	__m128 tms = _mm_setzero_ps();
	
	for(int i=0;i<num;i+=4)
	{
		__m128 ms = _mm_load_ps(src+i);
		tms = _mm_add_ps(tms,ms);
	}
	float data[4];
	_mm_storeu_ps(data,tms);
	return (data[0]+data[1]+data[2]+data[3]);
}

void Gaussianfilter(float* src, float* dest, int w, int h, int r)
{
	;
}

void Sobelfilter(float* src, float* dest, int w, int h, int r)
{
	;
}

void forkjoin_ex(float* src, float* dest0, float* dest1, float* dest2, int w, int h, int r)
{
	Gaussianfilter(src,dest0,w,h,r);
	Sobelfilter(src,dest1,w,h,r);
	boxfilter(src,dest0,w,h,r);
}

void forkjoin_ex_omp(float* src, float* dest0, float* dest1, float* dest2, int w, int h, int r)
{
#pragma omp parallel sections
	{
#pragma omp section
		{
			Gaussianfilter(src,dest0,w,h,r);
		}
#pragma omp section
		{
			Sobelfilter(src,dest1,w,h,r);
		}
#pragma omp section
		{
			boxfilter(src,dest0,w,h,r);
		}
	}
}