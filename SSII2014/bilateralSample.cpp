#include "function.h"

void bilateralFilterNaive(Mat& src, Mat& dest, int d, double sigma_color, double sigma_space)
{
	Mat srcd;src.convertTo(srcd,CV_64F);
	Mat destd = Mat::zeros(src.size(),CV_64F);
	const int r = d/2;
	for(int j=0;j<src.rows;j++)
	{
		for(int i=0;i<src.cols;i++)
		{
			double sum = 0.0;
			double coeff = 0.0;
			const double cp = srcd.at<double>(j,i);
			for(int l=-r;l<=r;l++)
			{
				for(int k=-r;k<=r;k++)
				{
					if(sqrt(l*l+k*k)<=r && i+k>=0 && i+k<src.cols && j+l>=0 && j+l<src.rows )
					{
						double c = exp(-0.5*((srcd.at<double>(j+l,i+k)-cp)*(srcd.at<double>(j+l,i+k)-cp))/(sigma_color*sigma_color));
						double s = exp(-0.5*(l*l+k*k)/(sigma_space*sigma_space));
						coeff+=c*s;
						sum+=srcd.at<double>(j+l,i+k)*c*s;
					}
				}
			}
			destd.at<double>(j,i)=sum/coeff;
		}
	}
	destd.convertTo(dest,src.type());
}

class BilateralFilter_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, int _radiusH, int _radiusV, int _maxk,
		int* _space_ofs, float *_space_weight, float *_color_weight) :
	temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV),
		maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
	{
	}

	virtual void operator() (const Range& range) const
	{
		int i, j, k;
		int cn = dest->channels();
		Size size = dest->size();

#if CV_SSE4_1
		bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if( cn == 1 )
		{
			uchar CV_DECL_ALIGNED(16) buf[16];

			uchar* sptr = (uchar*)temp->ptr(range.start+radiusV) + 16 * (radiusH/16 + 1);
			uchar* dptr = dest->ptr(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;
			for(i = range.start; i != range.end; i++,dptr+=dstep,sptr+=sstep )
			{
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];

						float* spw = space_weight;

						const uchar* sptrj = sptr+j;

						const __m128i sval0 = _mm_load_si128((__m128i*)(sptrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 tval1 = _mm_set1_ps(0.0f);
						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 tval2 = _mm_set1_ps(0.0f);
						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 tval3 = _mm_set1_ps(0.0f);
						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 tval4 = _mm_set1_ps(0.0f);

						const __m128i zero = _mm_setzero_si128();
						for(k = 0;  k < maxk; k ++, ofs++,spw++)
						{
							__m128i sref = _mm_loadu_si128((__m128i*)(sptrj+*ofs));
							_mm_store_si128((__m128i*)buf,_mm_add_epi8(_mm_subs_epu8(sval0,sref),_mm_subs_epu8(sref,sval0)));

							__m128i m1 = _mm_unpacklo_epi8(sref,zero);
							__m128i m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);

							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));
							__m128 _valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1,_valF);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2,_valF);
							wval2 = _mm_add_ps(wval2,_w);

							m1 = _mm_unpackhi_epi8(sref,zero);
							m2 = _mm_unpackhi_epi16(m1,zero);
							m1 = _mm_unpacklo_epi16(m1,zero);


							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));
							_valF =_mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3,_w);
							tval3 = _mm_add_ps(tval3,_valF);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));
							_valF =_mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4,_w);
							tval4 = _mm_add_ps(tval4,_valF);
						}
						tval1 = _mm_div_ps(tval1,wval1);
						tval2 = _mm_div_ps(tval2,wval2);
						tval3 = _mm_div_ps(tval3,wval3);
						tval4 = _mm_div_ps(tval4,wval4);
						_mm_stream_si128((__m128i*)(dptr+j), _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)) , _mm_packs_epi32( _mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar val0 = sptr[0];
					float sum=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int val = sptr[j + space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(val - val0)];
						sum += val*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U
					dptr[j] = (uchar)cvRound(sum/wsum);
				}
			}
		}
		else
		{
			short CV_DECL_ALIGNED(16) buf[16];

			const int sstep = 3*temp->cols;
			const int dstep = dest->cols*3;

			uchar* sptrr = (uchar*)temp->ptr(3*radiusV+3*range.start  ) + 16 * (radiusH/16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3*radiusV+3*range.start+1) + 16 * (radiusH/16 + 1);
			uchar* sptrb = (uchar*)temp->ptr(3*radiusV+3*range.start+2) + 16 * (radiusH/16 + 1);

			uchar* dptr = dest->ptr(range.start);			

			for(i = range.start; i != range.end; i++,sptrr+=sstep,sptrg+=sstep,sptrb+=sstep,dptr+=dstep )
			{	
				j=0;
#if CV_SSE4_1
				if( haveSSE4 )
				{
					for(; j < size.width; j+=16)//16 pixel unit
					{
						int* ofs = &space_ofs[0];

						float* spw = space_weight;

						const uchar* sptrrj = sptrr+j;
						const uchar* sptrgj = sptrg+j;
						const uchar* sptrbj = sptrb+j;

						const __m128i bval0 = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval0 = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval0 = _mm_load_si128((__m128i*)(sptrrj));

						__m128 wval1 = _mm_set1_ps(0.0f);
						__m128 rval1 = _mm_set1_ps(0.0f);
						__m128 gval1 = _mm_set1_ps(0.0f);
						__m128 bval1 = _mm_set1_ps(0.0f);

						__m128 wval2 = _mm_set1_ps(0.0f);
						__m128 rval2 = _mm_set1_ps(0.0f);
						__m128 gval2 = _mm_set1_ps(0.0f);
						__m128 bval2 = _mm_set1_ps(0.0f);

						__m128 wval3 = _mm_set1_ps(0.0f);
						__m128 rval3 = _mm_set1_ps(0.0f);
						__m128 gval3 = _mm_set1_ps(0.0f);
						__m128 bval3 = _mm_set1_ps(0.0f);

						__m128 wval4 = _mm_set1_ps(0.0f);
						__m128 rval4 = _mm_set1_ps(0.0f);
						__m128 gval4 = _mm_set1_ps(0.0f);
						__m128 bval4 = _mm_set1_ps(0.0f);

						const __m128i zero = _mm_setzero_si128();

						for(k = 0;  k < maxk; k ++, ofs++, spw++)
						{
							__m128i bref = _mm_loadu_si128((__m128i*)(sptrbj+*ofs));
							__m128i gref = _mm_loadu_si128((__m128i*)(sptrgj+*ofs));
							__m128i rref = _mm_loadu_si128((__m128i*)(sptrrj+*ofs));

							__m128i r1 = _mm_add_epi8(_mm_subs_epu8(rval0,rref),_mm_subs_epu8(rref,rval0));
							__m128i r2 = _mm_unpackhi_epi8(r1,zero);
							r1 = _mm_unpacklo_epi8(r1,zero);

							__m128i g1 = _mm_add_epi8(_mm_subs_epu8(gval0,gref),_mm_subs_epu8(gref,gval0));
							__m128i g2 = _mm_unpackhi_epi8(g1,zero);
							g1 = _mm_unpacklo_epi8(g1,zero);

							r1 = _mm_add_epi16(r1,g1);
							r2 = _mm_add_epi16(r2,g2);

							__m128i b1 = _mm_add_epi8(_mm_subs_epu8(bval0,bref),_mm_subs_epu8(bref,bval0));
							__m128i b2 = _mm_unpackhi_epi8(b1,zero);
							b1 = _mm_unpacklo_epi8(b1,zero);

							r1 = _mm_add_epi16(r1,b1);
							r2 = _mm_add_epi16(r2,b2);

							_mm_store_si128((__m128i*)(buf+8),r2);
							_mm_store_si128((__m128i*)buf,r1);

							r1 = _mm_unpacklo_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);
							g1 = _mm_unpacklo_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);
							b1 = _mm_unpacklo_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							const __m128 _sw = _mm_set1_ps(*spw);
							__m128 _w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[3]],color_weight[buf[2]],color_weight[buf[1]],color_weight[buf[0]]));

							__m128 _valr = _mm_cvtepi32_ps(r1);
							__m128 _valg = _mm_cvtepi32_ps(g1);
							__m128 _valb = _mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval1 = _mm_add_ps(rval1,_valr);
							gval1 = _mm_add_ps(gval1,_valg);
							bval1 = _mm_add_ps(bval1,_valb);
							wval1 = _mm_add_ps(wval1,_w);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[7]],color_weight[buf[6]],color_weight[buf[5]],color_weight[buf[4]]));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							rval2 = _mm_add_ps(rval2,_valr);
							gval2 = _mm_add_ps(gval2,_valg);
							bval2 = _mm_add_ps(bval2,_valb);
							wval2 = _mm_add_ps(wval2,_w);

							r1 = _mm_unpackhi_epi8(rref,zero);
							r2 = _mm_unpackhi_epi16(r1,zero);
							r1 = _mm_unpacklo_epi16(r1,zero);

							g1 = _mm_unpackhi_epi8(gref,zero);
							g2 = _mm_unpackhi_epi16(g1,zero);
							g1 = _mm_unpacklo_epi16(g1,zero);

							b1 = _mm_unpackhi_epi8(bref,zero);
							b2 = _mm_unpackhi_epi16(b1,zero);
							b1 = _mm_unpacklo_epi16(b1,zero);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[11]],color_weight[buf[10]],color_weight[buf[9]],color_weight[buf[8]]));

							_valr =_mm_cvtepi32_ps(r1);
							_valg =_mm_cvtepi32_ps(g1);
							_valb =_mm_cvtepi32_ps(b1);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval3 = _mm_add_ps(wval3,_w);
							rval3 = _mm_add_ps(rval3,_valr);
							gval3 = _mm_add_ps(gval3,_valg);
							bval3 = _mm_add_ps(bval3,_valb);

							_w = _mm_mul_ps(_sw,_mm_set_ps(color_weight[buf[15]],color_weight[buf[14]],color_weight[buf[13]],color_weight[buf[12]]));

							_valr =_mm_cvtepi32_ps(r2);
							_valg =_mm_cvtepi32_ps(g2);
							_valb =_mm_cvtepi32_ps(b2);

							_valr = _mm_mul_ps(_w, _valr);
							_valg = _mm_mul_ps(_w, _valg);
							_valb = _mm_mul_ps(_w, _valb);

							wval4 = _mm_add_ps(wval4,_w);
							rval4 = _mm_add_ps(rval4,_valr);
							gval4 = _mm_add_ps(gval4,_valg);
							bval4 = _mm_add_ps(bval4,_valb);
						}

						rval1 = _mm_div_ps(rval1,wval1);
						rval2 = _mm_div_ps(rval2,wval2);
						rval3 = _mm_div_ps(rval3,wval3);
						rval4 = _mm_div_ps(rval4,wval4);
						__m128i a = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)) , _mm_packs_epi32( _mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));
						gval1 = _mm_div_ps(gval1,wval1);
						gval2 = _mm_div_ps(gval2,wval2);
						gval3 = _mm_div_ps(gval3,wval3);
						gval4 = _mm_div_ps(gval4,wval4);
						__m128i b = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)) , _mm_packs_epi32( _mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						bval1 = _mm_div_ps(bval1,wval1);
						bval2 = _mm_div_ps(bval2,wval2);
						bval3 = _mm_div_ps(bval3,wval3);
						bval4 = _mm_div_ps(bval4,wval4);
						__m128i c = _mm_packus_epi16(_mm_packs_epi32( _mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)) , _mm_packs_epi32( _mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));

						//sse4///


						const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
						const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
						const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

						const __m128i bmask1 = _mm_setr_epi8
							(0,255,255,0,255,255,0,255,255,0,255,255,0,255,255,0);

						const __m128i bmask2 = _mm_setr_epi8
							(255,255,0,255,255,0,255,255,0,255,255,0,255,255,0,255);

						a = _mm_shuffle_epi8(a,mask1);
						b = _mm_shuffle_epi8(b,mask2);
						c = _mm_shuffle_epi8(c,mask3);
						uchar* dptrc = dptr+3*j;
						_mm_stream_si128((__m128i*)(dptrc),_mm_blendv_epi8(c,_mm_blendv_epi8(a,b,bmask1),bmask2));
						_mm_stream_si128((__m128i*)(dptrc+16),_mm_blendv_epi8(b,_mm_blendv_epi8(a,c,bmask2),bmask1));		
						_mm_stream_si128((__m128i*)(dptrc+32),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask2),bmask1));
					}
				}
#endif
				for(; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr+j;
					const uchar* sptrgj = sptrg+j;
					const uchar* sptrbj = sptrb+j;

					int r0 = sptrrj[0];
					int g0 = sptrgj[0];
					int b0 = sptrbj[0];

					float sum_r=0.0f,sum_b=0.0f,sum_g=0.0f;
					float wsum=0.0f;
					for(k=0 ; k < maxk; k++ )
					{
						int r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						float w = space_weight[k]*color_weight[std::abs(b - b0) +std::abs(g - g0) + std::abs(r - r0)];
						sum_b += b*w;
						sum_g += g*w;
						sum_r += r*w;
						wsum += w;
					}
					//overflow is not possible here => there is no need to use CV_CAST_8U

					wsum = 1.f/wsum;
					b0 = cvRound(sum_b*wsum);
					g0 = cvRound(sum_g*wsum);
					r0 = cvRound(sum_r*wsum);
					dptr[3*j] = (uchar)r0; dptr[3*j+1] = (uchar)g0; dptr[3*j+2] = (uchar)b0;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk, *space_ofs;
	float *space_weight, *color_weight;
};




//8u
void splitBGRLineInterleave_8u( const Mat& src, Mat& dest)
{

	const int size = src.size().area();
	dest.create(Size(src.cols,src.rows*3),CV_8U);
	const int dstep = src.cols*3;
	const int sstep = src.cols*3;

	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(1);
	uchar* R = dest.ptr<uchar>(2);

	//BGR BGR BGR BGR BGR B
	//GR BGR BGR BGR BGR BG
	//R BGR BGR BGR BGR BGR
	//BBBBBBGGGGGRRRRR shuffle
	const __m128i mask1 = _mm_setr_epi8(0,3,6,9,12,15,1,4,7,10,13,2,5,8,11,14);
	//GGGGGBBBBBBRRRRR shuffle
	const __m128i smask1 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
	const __m128i ssmask1 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

	//GGGGGGBBBBBRRRRR shuffle
	const __m128i mask2 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);
	//const __m128i smask2 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
	const __m128i ssmask2 = _mm_setr_epi8(0,1,2,3,4,11,12,13,14,15,5,6,7,8,9,10);

	//RRRRRRGGGGGBBBBB shuffle -> same mask2
	//__m128i mask3 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);

	//const __m128i smask3 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,6,7,8,9,10);
	//const __m128i ssmask3 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

	const __m128i bmask1 = _mm_setr_epi8
		(255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0);

	const __m128i bmask2 = _mm_setr_epi8
		(255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0);

	const __m128i bmask3 = _mm_setr_epi8
		(255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0);

	const __m128i bmask4 = _mm_setr_epi8
		(255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0);	

	__m128i a,b,c;

	for(int j=0;j<src.rows;j++)
	{
		int i=0;
		for(;i<src.cols;i+=16)
		{
			a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i)),mask1);
			b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i+16)),mask2);
			c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s+3*i+32)),mask2);
			_mm_stream_si128((__m128i*)(B+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask1),bmask2));

			a = _mm_shuffle_epi8(a,smask1);
			b = _mm_shuffle_epi8(b,smask1);
			c = _mm_shuffle_epi8(c,ssmask1);
			_mm_stream_si128((__m128i*)(G+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask2));

			a = _mm_shuffle_epi8(a,ssmask1);
			c = _mm_shuffle_epi8(c,ssmask1);
			b = _mm_shuffle_epi8(b,ssmask2);

			_mm_stream_si128((__m128i*)(R+i),_mm_blendv_epi8(c,_mm_blendv_epi8(b,a,bmask3),bmask4));
		}
		R+=dstep;
		G+=dstep;
		B+=dstep;
		s+=sstep;
	}
}

void splitBGRLineInterleave_32f( const Mat& src, Mat& dest)
{

	const int size = src.size().area();
	dest.create(Size(src.cols,src.rows*3),CV_32F);
	const int dstep = src.cols*3;
	const int sstep = src.cols*3;

	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(1);
	float* R = dest.ptr<float>(2);

	for(int j=0;j<src.rows;j++)
	{
		int i=0;
		for(;i<src.cols;i+=4)
		{
			__m128 a = _mm_load_ps((s+3*i));
			__m128 b = _mm_load_ps((s+3*i+4));
			__m128 c = _mm_load_ps((s+3*i+8));

			__m128 aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,2,3,0));
			aa=_mm_blend_ps(aa,b,4);
			__m128 cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(1,3,2,0));
			aa=_mm_blend_ps(aa,cc,8);
			_mm_stream_ps((B+i),aa);

			aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,2,0,1));
			__m128 bb = _mm_shuffle_ps(b,b,_MM_SHUFFLE(2,3,0,1));
			bb=_mm_blend_ps(bb,aa,1);
			cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(2,3,1,0));
			bb=_mm_blend_ps(bb,cc,8);
			_mm_stream_ps((G+i),bb);

			aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
			bb=_mm_blend_ps(aa,b,2);
			cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(3,0,1,2));
			cc=_mm_blend_ps(bb,cc,12);
			_mm_stream_ps((R+i),cc);

		}
		R+=dstep;
		G+=dstep;
		B+=dstep;
		s+=sstep;
	}
}
void splitBGRLineInterleave( const Mat& src, Mat& dest)
{
	if(src.type()==CV_MAKE_TYPE(CV_8U,3))
	{
		splitBGRLineInterleave_8u(src,dest);
	}
	else if(src.type()==CV_MAKE_TYPE(CV_32F,3))
	{
		splitBGRLineInterleave_32f(src,dest);
	}
}
void mybilateralFilter( const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int borderType )
{
	if(kernelSize.width==0 || kernelSize.height==0){ src.copyTo(dst);return;}
	int cn = src.channels();
	int i, j, maxk;
	Size size = src.size();

	CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) &&
		src.type() == dst.type() && src.size() == dst.size());

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);


	int radiusH = kernelSize.width>>1;
	int radiusV = kernelSize.height>>1;

	Mat temp;

	int dpad = (16- src.cols%16)%16;
	int spad =  dpad + (16-(2*radiusH)%16)%16;
	if(spad<16) spad +=16;
	int lpad = 16*(radiusH/16+1)-radiusH;
	int rpad = spad-lpad;
	if(cn==1)
	{
		copyMakeBorder( src, temp, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
	}
	else if (cn==3)
	{
		Mat temp2;
		copyMakeBorder( src, temp2, radiusV, radiusV, radiusH+lpad, radiusH+rpad, borderType );
		splitBGRLineInterleave(temp2,temp);
	}

	/*double minv,maxv;
	minMaxLoc(src,&minv,&maxv);
	const int color_range = cvRound(maxv-minv);*/
	const int color_range=256;
	vector<float> _color_weight(cn*color_range);

	//float CV_DECL_ALIGNED(16) _space_weight[255];
	vector<float> _space_weight(kernelSize.area()+1);
	vector<int> _space_ofs(kernelSize.area()+1);
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	// initialize color-related bilateral filter coefficients

	for( i = 0; i < color_range*cn; i++ )
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for( i = -radiusV, maxk = 0; i <= radiusV; i++ )
	{
		for(j = -radiusH ;j <= radiusH; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > max(radiusV,radiusH) )
				continue;
			space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
			space_ofs[maxk++] = (int)(i*temp.cols*cn + j);
		}
	}

	Mat dest = Mat::zeros(Size(src.cols+dpad, src.rows),dst.type());
	BilateralFilter_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, color_weight);
	parallel_for_(Range(0, size.height), body);
	Mat(dest(Rect(0,0,dst.cols,dst.rows))).copyTo(dst);
}

void opencvtest()
{
	Mat gray = imread("kodim23.png",0);
	Mat dest;

	for(int r=1;r<51;r+=5)
	{
		cout<<"radias: "<<r<<endl;
		int d = r*2+1;
		timer t;
		const int iter = 1;

		cout<<"naive"<<endl;
		t.start();
		for(int i=0;i<iter;i++)
			bilateralFilterNaive(gray,dest, d,30,30);
		t.stop();

		cout<<"opencv"<<endl;
		t.start();
		for(int i=0;i<iter;i++)
			bilateralFilter(gray,dest, d,30,30);
		t.stop();

		cout<<"my birateral"<<endl;
		t.start();
		for(int i=0;i<iter;i++)
			mybilateralFilter(gray,dest, Size(d,d),30,30,4);
		t.stop();
		cout<<endl;
	}
}