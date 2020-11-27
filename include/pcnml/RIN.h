//
// Created by bohuan on 2019/10/18.
//

#ifndef PCNML_RIN_H_
#define PCNML_RIN_H_
//主程序，用来复现
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#include <unistd.h>
#include <chrono>

#include <x86intrin.h>

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <VCL/vectorclass.h>
#include <iostream>

#include "utils/visualization.h"
#include <avxintrin.h>
#include <avx2intrin.h>

//#include <linasm/Math.h>
//#include <linasm/Math.h>
#include <sse_mathfun.h>

#define EPS 1e-7
#define FEQU_ZERO(x)  (  -EPS<(x) && (x)<EPS)


#define T256(X) (*((__m256*)&(X)))
#define T128(X) (*((__m128*)&(X)))

void TESTBUILD(){

}

static inline float FastArcTan(float x)
{
  //std::cout <<x<<" "<<  (atan(x) - (M_PI_4*x - x*(fabs(x) - 1)*(0.2447 + 0.0663*fabs(x)))) << std::endl;
  return M_PI_4*x - x*(fabs(x) - 1)*(0.2447 + 0.0663*fabs(x));
}

inline void atan(Vec8f &x){
  float *p = (float*)(&x);
  *(p++) = atan(*(p));
  *(p++) = atan(*(p));
  *(p++) = atan(*(p));
  *(p++) = atan(*(p));

  *(p++) = atan(*(p));
  *(p++) = atan(*(p));
  *(p++) = atan(*(p));
  *(p) = atan(*(p));
}

inline Vec8f sin(Vec8f &x){
  Vec8f ret;

  *((Vec4f*)&ret) = sin_ps(*((Vec4f*)&x));
  *(((Vec4f*)&ret)+1) = sin_ps(*(((Vec4f*)&x)+1));
  return ret;
}

inline Vec8f cos(Vec8f &x){
  Vec8f ret;
  *((Vec4f*)&ret) = cos_ps(*((Vec4f*)&x));
  *(((Vec4f*)&ret)+1) = cos_ps(*(((Vec4f*)&x)+1));
  return ret;
}


inline void exp_slow(Vec8f &x) {
  *((Vec4f*)&x) = cos_ps(*((Vec4f*)&x));
  *(((Vec4f*)&x)+1) = cos_ps(*(((Vec4f*)&x)+1));
}


inline void exp(Vec8f &x){
  const Vec8f ONE(1.0);
  const Vec8f INV256(1.0/256.0);
  x = ONE + x * INV256;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
}

#define TT(Z) Z(v-1, u-1),\
         Z(v-1,u),\
         Z(v-1,u + 1),\
         Z(v, u - 1),\
         Z(v, u + 1),\
         Z(v + 1, u - 1),\
         Z(v + 1, u),\
         Z(v + 1, u + 1)


#define M256_GRID(x) \
    *(idx_##x##_m1 - 1),\
    *(idx_##x##_m1),\
    *(idx_##x##_m1+1),\
    *(idx_##x - 1),\
    *(idx_##x + 1),\
    *(idx_##x##_p1 -1),\
    *(idx_##x##_p1),\
    *(idx_##x##_p1 + 1)



enum RIN_METHOD{R_MEANS_8, //8点均值法，比较快，精准低一些.
  R_MEDIAN_FAST_8,        //8点快速中值，时间和精度介于 8点均值和8点完整中值之间.
  R_MEDIAN_STABLE_8,      //8点稳定中值，时间慢，精度高.
  R_MEANS_4,              //4点均值， 速度快，精度高.
  R_MEDIAN_4,             //4点中值， 速度稍快，精度一般.(目前来看不推荐，没啥价值).
  R_MEDIAN_FAST_4_8,           //用4个点求梯度，但是算中值用周围8个点.
  R_MEDIAN_STABLE_4_8,           //用4个点求梯度，但是算中值用周围8个点.
  R_MEANS_4_8,
  R_MEANS_SOBEL,
  R_MEDIAN_SOBEL,
  R_MEANS_SCHARR,
  R_MEDIAN_SCHARR,
  R_MEANS_PREWITT,
  R_MEDIAN_PREWITT};

/**
 * @brief 给3个cvmat然后把数据归到cv::mat然后输出,
 * 有可能nx,ny,nz的大小和实际的cols,和rows不一样，所以要给参数
 * @param nx X轴的normal
 * @param ny Y轴的normal
 * @param nz Z轴的normal
 * @param cols 图像的列
 * @param rows 图像的行 */
static inline cv::Mat PCNML_normalizatio(cv::Mat &nx,
                                         cv::Mat &ny, cv::Mat &nz,
                                         const int cols, const int rows){

  cv::Mat result(cv::Size(cols, rows), CV_32FC3);
  int u, v;
  for (u = 1; u < result.cols - 1; ++ u)
    for (v = 1; v < result.rows - 1; ++ v){
      cv::Vec3f x = {nx.at<float>(v, u), ny.at<float>(v, u), nz.at<float>(v, u)};
      float d = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));
      x(0) /= d;
      x(1) /= d;
      x(2) /= d;
      if (x(2)>0){
        x(0) = - x(0);
        x(1) = - x(1);
        x(2) = - x(2);
      }
      result.at<cv::Vec3f>(v, u) = x;
    }
  return result;

}

/**
 * @brief ...
 * @param[in] matpart 分别保存x,y,z的数据
 * @param[in] camera 相机参数3*3矩阵K，你们懂的
 * @param[in] method 调用方法,目前只支持均值法，因为最快
 * @param[in] need_normal 结果是否要归一化
 * @return 返回normal图片 */
static inline cv::Mat PCNML(const std::vector<cv::Mat> &matpart,
                            const cv::Matx33d camera,
                            const RIN_METHOD method,
                            const bool need_normal){
  //double st = clock();
  cv::Mat d = 1/matpart.at(2);
  cv::Mat Gv, Gu;
  //cv::Mat sobel_y = (cv::Mat_<float>(3, 3)<< -1, -2, -1, 0, 0, 0, 1, 2, 1);
  //cv::Mat sobel_x = (cv::Mat_<float>(3, 3)<< -1, 0, 1, -2, 0, 2, -1, 0, 1);
  //cv::filter2D(d, Gv, CV_32F, -sobel_y);
  //cv::filter2D(d, Gu, CV_32F, -sobel_x);

  cv::Sobel(d, Gv, CV_32F, 0, 1, 3, -camera(0,0)); //y方向
  cv::Sobel(d, Gu, CV_32F, 1, 0, 3, -camera(1,1)); //x方向

  cv::Mat_<float> X(matpart.at(0));
  cv::Mat_<float> Y(matpart.at(1));
  cv::Mat_<float> Z(matpart.at(2));
  cv::Mat_<float> D(d);

  cv::Mat nx_t = Gu;/// * K(1,1);
  cv::Mat ny_t = Gv;// * K(2, 2);
  cv::Mat nz_t(d);
  cv::Mat_<float> nx(nx_t);
  cv::Mat_<float> ny(ny_t);
  cv::Mat_<float> nz(nz_t);


  //显然按照经验v u比较快
  register_t u, v, xr(X.rows - 1), xc(X.cols -1);

  const Vec8f ZERO(1e5);
  const Vec8f NEG_INF(-1e5);
  const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};

  Vec8f DX;
  Vec8f DY;
  Vec8f DZ;

  auto COL = d.cols;//列数

  //m1 就是减1行，p1就是加一行
  float *idx_nx((float*)nx_t.data + COL);// *idx_nx_m1((float*)nx_t.data-COL), *idx_nx_p1((float*)nx_t.data + COL);
  float *idx_ny((float*)ny_t.data + COL);// *idx_ny_m1((float*)ny_t.data-COL), *idx_ny_p1((float*)ny_t.data + COL);
  float *idx_nz((float*)nz_t.data + COL);// *idx_nz_m1((float*)nz_t.data-COL), *idx_nz_p1((float*)nz_t.data + COL);

  float *idx_x((float*)X.data+COL), *idx_x_m1((float*)X.data), *idx_x_p1((float*)X.data + 2*COL);
  float *idx_y((float*)Y.data+COL), *idx_y_m1((float*)Y.data), *idx_y_p1((float*)Y.data + 2*COL);
  float *idx_z((float*)Z.data+COL), *idx_z_m1((float*)Z.data), *idx_z_p1((float*)Z.data + 2*COL);
  float *end_nx;
  int ONE_MINUS_COL = COL - 1;


#define UPDATE_IDX idx_nx++, idx_ny++, idx_nz++, idx_x++, idx_y++, idx_z++, idx_x_m1++, idx_y_m1++, idx_z_m1++, idx_x_p1++, idx_y_p1++,idx_z_p1++
  for (v = 1; v < xr; ++ v){
    end_nx = idx_nx + ONE_MINUS_COL;
    UPDATE_IDX;
    for (;idx_nx != end_nx; UPDATE_IDX) {
      if ((*((int*)idx_nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)idx_ny) & 0x7FFFFFFF) < 1e-7){
        *idx_nz = -1;
        continue;
      }
      DX = ((*idx_x) - Vec8f(M256_GRID(x))) * (*idx_nx);
      DY = ((*idx_y) - Vec8f(M256_GRID(y))) * (*idx_ny);
      DZ = Vec8f(M256_GRID(z)) - (*idx_z) ;
      DX = (DX+DY) / DZ;

      Vec8fb LT_ZERO = _mm256_cmp_ps(DX, ZERO, _CMP_LE_OS);
      Vec8fb LG_NG_INF = _mm256_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
      Vec8fb BOOL_RESULT = _mm256_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
      Vec8i x=_mm256_srli_epi32(_mm256_castps_si256(BOOL_RESULT), 31);
      float sum = horizontal_add((Vec8f)(_mm256_and_ps(T256(BOOL_RESULT), T256(DX))));
      auto n = horizontal_add(x);
      *idx_nz = sum * mul[n];
    }
    UPDATE_IDX;
  }
#undef UPDATE_IDX
  //std::cout<<"the time is : " << (clock()-st)/CLOCKS_PER_SEC << std::endl;
  return PCNML_normalizatio(nx_t, ny_t, nz_t, nx_t.cols, nx_t.rows);
}


#define SURREND(x) idx_m1[-1][x], idx_m1[0][x], idx_m1[1][x],idx[-1][x], idx[1][x],idx_p1[-1][x], idx_p1[0][x], idx_p1[1][x]
/**
 * @brief 中间是0的3*3的kernal来做我们的算法。
 * 一定要狐疑 kernal的中间是0
 * @param kernal_x 因为中间是0，所以分别为
 * abc
 * d?e
 * fgh
 * 这样形状的kernal,按照abcd?efgh，的顺序，其中?是0，就不用保存了。
 * @param kernal_y 累死kernal_x
 * @param camera 相机参数
 * @param input 输入的RGB(XYZ)的图像
 * @param output 输出经过filter某个channal
 * */
static inline void PCNML_MEAN(const Vec8f &kernal_x, const Vec8f &kernal_y,
                              //const cv::Matx33d &camera,
                              const cv::Mat &input,
                              cv::Mat &output){

  output.create(input.rows,input.cols, CV_32FC3);//3 通道，非常make sense
  //x是竖着，y是横着的
  int COL = input.cols;
  cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
  cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
  cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
  cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
  cv::Vec3f* end_idx;

  int ONE_MINUS_COL = COL - 1;
  int v(2);
  const Vec8f ZERO(1e5);
  const Vec8f NEG_INF(-1e5);
  const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
  for (; v!=input.rows; ++v){
    end_idx = idx + ONE_MINUS_COL;
    UPDATE_IDX;
    for (; idx != end_idx;UPDATE_IDX){
      Vec8f D = Vec8f(SURREND(2));
      float& nx = idx_o->operator()(0)=horizontal_add(kernal_x / D);
      float& ny = idx_o->operator()(1)=horizontal_add(kernal_y / D);

      if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
        idx_o->operator()(2) = -1;
        continue;
      }
      Vec8f DX =  (idx->operator()(0) - Vec8f(SURREND(0))) * nx;
      Vec8f DY =  (idx->operator()(1) - Vec8f(SURREND(1))) * ny;

      float &x = idx->operator()(0);
      float &y = idx->operator()(1);
      //Vec8f Z = Vec8f(SURREND(2)) - idx->operator()(2);
      Vec8f Z = D - idx->operator()(2);
      DX = (DX + DY) / Z;

      Vec8fb LT_ZERO = _mm256_cmp_ps(DX, ZERO, _CMP_LE_OS);
      Vec8fb LG_NG_INF = _mm256_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
      Vec8fb BOOL_RESULT = _mm256_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
      Vec8i number=_mm256_srli_epi32(_mm256_castps_si256(BOOL_RESULT), 31);
      float sum = horizontal_add((Vec8f)(_mm256_and_ps(T256(BOOL_RESULT), T256(DX))));
      auto n = horizontal_add(number);
      idx_o->operator()(2) = sum * mul[n];
    }
    UPDATE_IDX;
  }
}

//TODO 现在不快
static inline void PCNML_MEDIAN_FAST(const Vec8f &kernal_x, const Vec8f &kernal_y,
                                     const cv::Matx33d &camera,
                                     const cv::Mat &input,
                                     cv::Mat &output){

  output.create(input.rows,input.cols, CV_32FC3);//3 通道，非常make sense
  //x是竖着，y是横着的
  int COL = input.cols;
  cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
  cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
  cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
  cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
  cv::Vec3f* end_idx;

  int ONE_MINUS_COL = COL - 1;
  int v(2);
  const Vec8f ZERO(1e5);
  const Vec8f NEG_INF(-1e5);
  const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
  for (; v!=input.rows; ++v){
    end_idx = idx + ONE_MINUS_COL;
    UPDATE_IDX;
    for (; idx != end_idx;UPDATE_IDX){
      Vec8f D = Vec8f(SURREND(2));
      float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
      float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

      if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
        idx_o->operator()(2) = -1;
        continue;
      }
      Vec8f DX =  (idx->operator()(0) - Vec8f(SURREND(0))) * nx;
      Vec8f DY =  (idx->operator()(1) - Vec8f(SURREND(1))) * ny;

      float &x = idx->operator()(0);
      float &y = idx->operator()(1);
      Vec8f Z = D - idx->operator()(2);
      DX = (DX + DY) / Z;
      float *tmp=((float*)(&DX));
      int c=0;
      for (int i = 7; i >= 0; isnan(tmp[i])? i--: tmp[c++] = tmp[i--]);
      std::sort(tmp, tmp+c);
      if (c)  idx_o->operator()(2) = (c&1) ? tmp[c>>1] : (tmp[c>>1] + tmp[(c>>1)-1]) * 0.5;
      else idx_o->operator()(2) = 0;
      continue;
    }
    UPDATE_IDX;
  }
}

static inline void PCNML_MEDIAN_STABLE(const Vec8f &kernal_x, const Vec8f &kernal_y,
                                       const cv::Matx33d &camera,
                                       const cv::Mat &input,
                                       cv::Mat &output){

  output.create(input.rows,input.cols, CV_32FC3);//3 通道，非常make sense
  //x是竖着，y是横着的
  int COL = input.cols;
  cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
  cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
  cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
  cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
  cv::Vec3f* end_idx;

  int ONE_MINUS_COL = COL - 1;
  int v(2);
  const Vec8f ZERO(1e5);
  const Vec8f NEG_INF(-1e5);
  const float mul[9]={0,1,1/2.0,1/3.0,1/4.0,1/5.0,1/6.0,1/7.0,1/8.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
  for (; v!=input.rows; ++v){
    end_idx = idx + ONE_MINUS_COL;
    UPDATE_IDX;
    for (; idx != end_idx;UPDATE_IDX){
      Vec8f D = Vec8f(SURREND(2));
      float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
      float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

      if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
        idx_o->operator()(2) = -1;
        continue;
      }
      Vec8f DX =  (idx->operator()(0) - Vec8f(SURREND(0))) * nx;
      Vec8f DY =  (idx->operator()(1) - Vec8f(SURREND(1))) * ny;

      float &x = idx->operator()(0);
      float &y = idx->operator()(1);
      Vec8f Z = D - idx->operator()(2);
      DX = (DX + DY) / Z;
      float *tmp=((float*)(&DX));
      int c=0;
      for (int i = 7; i >= 0; isnan(tmp[i])? i--: tmp[c++] = tmp[i--]);
      std::sort(tmp, tmp+c);
      if (c)  idx_o->operator()(2) = (c&1) ? tmp[c>>1] : (tmp[c>>1] + tmp[(c>>1)-1]) * 0.5;
      else idx_o->operator()(2) = 0;
      continue;
    }
    UPDATE_IDX;
  }
}
#undef SURREND



#define SURREND(x) idx_m1[0][x], idx[-1][x], idx[1][x],idx_p1[0][x]
/**
 * @brief 和上面不同的是，kernal只有4个有效值，分别在上下左右*/
static inline void PCNML_MEAN(const Vec4f &kernal_x, const Vec4f &kernal_y,
                              const cv::Matx33d &camera,
                              const cv::Mat &input,
                              cv::Mat &output){

  output.create(input.rows,input.cols, CV_32FC3);//3 通道，非常make sense
  //x是竖着，y是横着的
  int COL = input.cols;
  cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
  cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
  cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
  cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
  cv::Vec3f* end_idx;

  int ONE_MINUS_COL = COL - 1;
  int v(2);
  const Vec4f ZERO(1e5);
  const Vec4f NEG_INF(-1e5);
  const float mul[5]={0,1,1/2.0,1/3.0,1/4.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
  for (; v!=input.rows; ++v){
    end_idx = idx + ONE_MINUS_COL;
    UPDATE_IDX;
    for (; idx != end_idx;UPDATE_IDX){
      Vec4f D = Vec4f(SURREND(2));
      float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
      float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

      if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
        idx_o->operator()(2) = -1;
        continue;
      }
      Vec4f DX =  (idx->operator()(0) - Vec4f(SURREND(0))) * nx;
      Vec4f DY =  (idx->operator()(1) - Vec4f(SURREND(1))) * ny;

      float &x = idx->operator()(0);
      float &y = idx->operator()(1);
      Vec4f Z = D - idx->operator()(2);
      DX = (DX + DY) / Z;

      Vec4fb LT_ZERO = _mm_cmp_ps(DX, ZERO, _CMP_LE_OS);
      Vec4fb LG_NG_INF = _mm_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
      Vec4fb BOOL_RESULT = _mm_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
      Vec4i number=_mm_srli_epi32(_mm_castps_si128(BOOL_RESULT), 31);
      float sum = horizontal_add((Vec4f)(_mm_and_ps(T128(BOOL_RESULT), T128(DX))));
      auto n = horizontal_add(number);
      idx_o->operator()(2) = sum * mul[n];
    }
    UPDATE_IDX;
  }
}


/**
 * @brief  版本
 * @param kernal_x 因为中间是0，所以分别为
 * abc
 * d?e
 * fgh
 * 这样形状的kernal,按照abcd?efgh，的顺序，其中?是0，就不用保存了。
 * @param kernal_y 累死kernal_x
 * @param camera 相机参数
 * @param input 输入的RGB(XYZ)的图像
 * @param output 输出经过filter某个channal
 * */
static inline void PCNML_FAST(const std::vector<cv::Mat> &matpart,
                            const cv::Matx33d camera,
                            cv::Mat *result){
  static bool is_init(false);
  static int rows, cols;
  if (!is_init){
    is_init = true;
    //设置好rows和cols的数据
  }




  cv::Mat_<float> X(matpart.at(0));
  cv::Mat_<float> Y(matpart.at(1));
  cv::Mat_<float> Z(matpart.at(2));
  cv::Mat d = 1 / Z;
  cv::Mat_<float> D(d);
  cv::Mat Gv, Gu;
  cv::Mat sobel_y = (cv::Mat_<float>(3, 3)<< 0 , -1, 0, 0, 0, 0, 0, 1, 0);
  cv::Mat sobel_x = (cv::Mat_<float>(3, 3)<< 0, 0, 0, -1, 0, 1, 0, 0, 0);
  cv::filter2D(D, Gv, CV_32F, -sobel_y);
  cv::filter2D(D, Gu, CV_32F, -sobel_x);

  //cv::Sobel(d, Gv, CV_32F, 0, 1, 3, -camera(0,0)); //y方向
  //cv::Sobel(d, Gu, CV_32F, 1, 0, 3, -camera(1,1)); //x方向
  cv::Mat mag = (Gv.mul(Gv) + Gu.mul(Gu));
  for (int i = 0; i < X.rows * X.cols; ++ i){
    mag.at<float>(i) = sqrt(mag.at<float>(i));
  }
  cv::Mat_<float> Mag = mag;

  cv::Mat nx_t = Gu * camera(0, 0);
  cv::Mat ny_t = Gv * camera(1, 1);
  //std::cout<<nx_t.at<float>(200,200)<<std::endl;

  double sigma = 20.5;

  const cv::Mat k1 = (cv::Mat_<float>(3,3) << -1, 0, 0,0,1,0,0,0,0);
  const cv::Mat k2 = (cv::Mat_<float>(3,3) <<  0, -1, 0,0,1,0,0,0,0);
  const cv::Mat k3 = (cv::Mat_<float>(3,3) <<  0, 0, -1,0,1,0,0,0,0);
  const cv::Mat k4 = (cv::Mat_<float>(3,3) <<  0, 0, 0,-1,1,0,0,0,0);
  const cv::Mat k5 = (cv::Mat_<float>(3,3) <<  0, 0, 0,0,1,-1,0,0,0);
  const cv::Mat k6 = (cv::Mat_<float>(3,3) <<  0, 0, 0,0,1,0,-1,0,0);
  const cv::Mat k7 = (cv::Mat_<float>(3,3) <<  0, 0, 0,0,1,0,0,-1,0);
  const cv::Mat k8 = (cv::Mat_<float>(3,3) <<  0, 0, 0,0,1,0,0,0,-1);

  const std::vector<cv::Mat> kernal = {k1, k2, k3, k4, k5, k6, k7, k8};
  cv::Mat x_d, y_d, z_d;
  cv::Mat map = cv::Mat::zeros(X.rows, X.cols, CV_32F);
  cv::Mat nz_j;
  cv::Mat nx_t_sum = cv::Mat::zeros(X.rows, X.cols, CV_32F);;
  cv::Mat ny_t_sum = cv::Mat::zeros(X.rows, X.cols, CV_32F);;
  cv::Mat nz_t_sum = cv::Mat::zeros(X.rows, X.cols, CV_32F);;

  for (int i = 0; i < 8; ++ i){
    cv::filter2D(X, x_d, CV_32F, -kernal.at(i));
    cv::filter2D(Y, y_d, CV_32F, -kernal.at(i));
    cv::filter2D(Z, z_d, CV_32F, -kernal.at(i));
    cv::Mat dist = x_d.mul(x_d) + y_d.mul(y_d) + z_d.mul(z_d);

    //cv::Mat w_map;
    for (int i = 0; i < X.rows * X.cols; ++ i){
      dist.at<float>(i) = sqrt(dist.at<float>(i));
    }
    dist = -(dist.mul(dist) / sigma);
    for (int i = 0; i < X.rows * X.cols; ++ i){
      dist.at<float>(i) = exp(dist.at<float>(i));
    }

    cv::Mat_<float> w_map = dist;
    nz_j = -(nx_t.mul(x_d) + ny_t.mul(y_d)) / z_d;
    map += w_map;

    cv::Mat tmp = nx_t.mul(nx_t) + ny_t.mul(ny_t) + nz_j.mul(nz_j);
    for (int i = 0; i < X.rows * X.cols; ++ i){
      tmp.at<float>(i) = sqrt(tmp.at<float>(i));
    }

    nx_t_sum += (nx_t / tmp).mul(w_map);
    ny_t_sum += (ny_t / tmp).mul(w_map);
    nz_t_sum += (nz_j / tmp).mul(w_map);

  }
  nx_t_sum = nx_t_sum / map;
  ny_t_sum = ny_t_sum / map;
  nz_t_sum = nz_t_sum / map;

  std::cout<<"!!"<<nx_t_sum.at<float>(200,200)<<std::endl;
  std::cout<<"!!"<<ny_t_sum.at<float>(200,200)<<std::endl;
  std::cout<<"!!"<<nz_t_sum.at<float>(200,200)<<std::endl;


  for (int i = 0; i < X.rows * X.cols; ++ i){
    float phi2 = atan(ny_t_sum.at<float>(i) / nx_t_sum.at<float>(i)) + M_PI;
    float phi1 = atan((nx_t_sum.at<float>(i) * cos(phi2) + ny_t_sum.at<float>(i)*sin(phi2))/nz_t_sum.at<float>(i));
    cv::Vec3f s(sin(phi1) * cos(phi2),
        sin(phi1) * sin(phi2),
        cos(phi1));
    result->at<cv::Vec3f>(i) = s;
  }

  //std::cout<<"!!"<<result->at<cv::Vec3f>(200,200)<<std::endl;






  //std::cout<<"the time is : " << (clock()-st)/CLOCKS_PER_SEC << std::endl;
//  return PCNML_normalizatio(nx_t, ny_t, nz_t, nx_t.cols, nx_t.rows);
}

//这个版本没用指令集加速
static inline void PCNML_LAST(const cv::Mat &input,
                              const cv::Matx33d camera,
                              cv::Mat &output){
  static bool is_init(false);
  static int rows, cols;
  if (!is_init){
    is_init = true;
    //设置好rows和cols的数据
  }

  output.create(input.rows, input.cols, CV_32FC3);

  int COL = input.cols;
  cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
  cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
  cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
  cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
  cv::Vec3f* end_idx;

  int ONE_MINUS_COL = COL - 1;
  int v(2);

  int debuga=0, debugb=0;
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
  for (; v!=input.rows; ++v){
    ++debuga;
    end_idx = idx + ONE_MINUS_COL;
    UPDATE_IDX;
    debugb=0;
    for (; idx != end_idx;UPDATE_IDX){
      ++debugb;
      float gv =  1.0 / idx_m1[0][2] - 1.0 / idx_p1[0][2];
      float gu =  1.0 / idx[-1][2] - 1.0 / idx[1][2];
     //float d = 1.0 / idx[0][2];
      //float mag = sqrt(gv * gv + gu * gu);
      float nx_t = gu * camera(0,0);
      float ny_t = gv * camera(1,1);


      const double sigma = 20.5;
      float nx_t_sum(0);
      float ny_t_sum(0);
      float nz_t_sum(0);
      float nz_j(0);
      float map(0);
      float x_d, y_d, z_d;

//      for (int i = 0; i <8; ++ i){
        const auto F=[&](){
          float w_map = exp(-fabs(x_d * x_d + y_d * y_d + z_d * z_d)/sigma);
          map += w_map;
          nz_j = -(nx_t * x_d + ny_t * y_d) / z_d;
          float tmp = sqrt(nx_t*nx_t + ny_t*ny_t + nz_j*nz_j);
          nx_t_sum += (nx_t / tmp) * w_map;
          ny_t_sum += (ny_t / tmp) * w_map;
          nz_t_sum += (nz_j / tmp) * w_map;
        };

      x_d = idx_m1[-1][0] - idx[0][0];
      y_d = idx_m1[-1][1] - idx[0][1];
      z_d = idx_m1[-1][2] - idx[0][2];
      F();

      x_d = idx_m1[0][0] - idx[0][0];
      y_d = idx_m1[0][1] - idx[0][1];
      z_d = idx_m1[0][2] - idx[0][2];
      F();
      x_d = idx_m1[1][0] - idx[0][0];
      y_d = idx_m1[1][1] - idx[0][1];
      z_d = idx_m1[1][2] - idx[0][2];
      F();

      x_d = idx[-1][0] - idx[0][0];
      y_d = idx[-1][1] - idx[0][1];
      z_d = idx[-1][2] - idx[0][2];
      F();

      x_d = idx[1][0] - idx[0][0];
      y_d = idx[1][1] - idx[0][1];
      z_d = idx[1][2] - idx[0][2];
      F();

      x_d = idx_p1[-1][0] - idx[0][0];
      y_d = idx_p1[-1][1] - idx[0][1];
      z_d = idx_p1[-1][2] - idx[0][2];
      F();

      x_d = idx_p1[0][0] - idx[0][0];
      y_d = idx_p1[0][1] - idx[0][1];
      z_d = idx_p1[0][2] - idx[0][2];
      F();
      x_d = idx_p1[1][0] - idx[0][0];
      y_d = idx_p1[1][1] - idx[0][1];
      z_d = idx_p1[1][2] - idx[0][2];
      F();

      nx_t_sum = nx_t_sum / map;
      ny_t_sum = ny_t_sum / map;
      nz_t_sum = nz_t_sum / map;

      //if (debuga==200 && debugb == 200){
      //  std::cout<<nx_t_sum<<std::endl;
      //  std::cout<<ny_t_sum<<std::endl;
      //  std::cout<<nz_t_sum<<std::endl;
      //}


      float phi2 = atan(ny_t_sum / nx_t_sum) + M_PI;
      float phi1 = atan((nx_t_sum * cos(phi2) + ny_t_sum*sin(phi2))/nz_t_sum);
      cv::Vec3f s(sin(phi1) * cos(phi2),
                  sin(phi1) * sin(phi2),
                  cos(phi1));
      *idx_o = s;
    }
    UPDATE_IDX;
  }
#undef UPDATE_IDX
  //std::cout<<"@"<<output.at<cv::Vec3f>(200,200) << std::endl;






  //std::cout<<"the time is : " << (clock()-st)/CLOCKS_PER_SEC << std::endl;
//  return PCNML_normalizatio(nx_t, ny_t, nz_t, nx_t.cols, nx_t.rows);
}

//指令集加速版本，对应paper:ICIP
static inline void PCNML_ICIP(const cv::Mat &input,
                              const cv::Matx33d camera,
                              cv::Mat &output){
  const double sigma = 20.5;
  const double inv_sigma = 1.0/20.5;
  static bool is_init(false);
  static int rows, cols;
  if (!is_init){
    is_init = true;
    //设置好rows和cols的数据
  }

  output.create(input.rows, input.cols, CV_32FC3);

  int COL = input.cols;

  Vec4f cam(camera(1,1), camera(1,1), camera(0,0), camera(0,0));
  cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
  cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
  cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
  cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
  cv::Vec3f* end_idx;
  const int ONE_MINUS_COL = COL - 1;
  int v(2);

  //int debuga=0, debugb=0;
  int count(0);
  cv::Vec3f * out[8];
  Vec8f nxtsum, nytsum, nztsum;


#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
  for (; v!=input.rows; ++v){
    //++debuga;
    end_idx = idx + ONE_MINUS_COL;
    UPDATE_IDX;
    //debugb=0;
    for (; idx != end_idx;UPDATE_IDX){
     // ++debugb;

      Vec4f gvu(idx_m1[0][2], idx_p1[0][2], idx[-1][2], idx[1][2]);
      gvu = cam / gvu;

      float nx_t = gvu[2] - gvu[3];
      float ny_t = gvu[0] - gvu[1];

      //float nx_t_sum(0);
      //float ny_t_sum(0);
      //float nz_t_sum(0);

      Vec8f tx(idx_m1[-1][0], idx_m1[0][0], idx_m1[1][0], idx[-1][0], idx[1][0], idx_p1[-1][0], idx_p1[0][0], idx_p1[1][0]);
      Vec8f ty(idx_m1[-1][1], idx_m1[0][1], idx_m1[1][1], idx[-1][1], idx[1][1], idx_p1[-1][1], idx_p1[0][1], idx_p1[1][1]);
      Vec8f tz(idx_m1[-1][2], idx_m1[0][2], idx_m1[1][2], idx[-1][2], idx[1][2], idx_p1[-1][2], idx_p1[0][2], idx_p1[1][2]);
      tx -= idx[0][0];
      ty -= idx[0][1];
      tz -= idx[0][2];
      Vec8f nzj = -(nx_t * tx + ny_t * ty) / tz;

      auto &tt = tx;//改名，用原来tx的内存
      tt = (tx + ty + tz)* inv_sigma;
      exp(tt);

      tt /= sqrt(nx_t * nx_t + ny_t * ny_t + nzj * nzj);
      float sum = horizontal_add(tt);

      ((float*)&nxtsum)[count] = sum * nx_t;
      ((float*)&nytsum)[count] = sum * ny_t;
      ((float*)&nztsum)[count] = horizontal_add(tt * nzj);

      out[count++] = idx_o;

      if (count == 8){
        count = 0;
        Vec8f phi2(M_PI);
        Vec8f tmp = nytsum / nxtsum;
        atan(tmp);
        phi2 += tmp;

        tmp = (nxtsum * cos(phi2) + nytsum * sin(phi2)) / nztsum;
        atan(tmp);
        Vec8f &phi1 = tmp;
        Vec8f X = sin(phi1) * cos(phi2);
        Vec8f Y = sin(phi1) * sin(phi2);
        Vec8f Z = cos(phi1);

        for (int i = 0; i < 8; ++ i) {
          if (isnan(X[i]) || isnan(Y[i]) || isnan(Z[i])){
            *out[i]= {0,0,-1};
          }
          else {
            *out[i] = {X[i], Y[i], Z[i]};
          }
        }

      }
      //std::cout<<"xx"<<std::endl;
      //   float phi2 = atan(ny_t_sum / nx_t_sum) + M_PI;
      //   float phi1 = atan((nx_t_sum * cos(phi2) + ny_t_sum*sin(phi2))/ nz_t_sum);

      //   cv::Vec3f s(sin(phi1) * cos(phi2),
      //               sin(phi1) * sin(phi2),
      //               cos(phi1));

      //   if (isnan(s[0]) || isnan(s[1]) || isnan(s[2])){
      //     s={0,0,-1};
      //   }
      //   *idx_o = s;

    }
    UPDATE_IDX;
  }
#undef UPDATE_IDX
  //std::cout<<"@"<<output.at<cv::Vec3f>(200,200) << std::endl;

  //std::cout<<"the time is : " << (clock()-st)/CLOCKS_PER_SEC << std::endl;
//  return PCNML_normalizatio(nx_t, ny_t, nz_t, nx_t.cols, nx_t.rows);
}


/**
 * @brief 和上面不同的是，kernal只有4个有效值，分别在上下左右*/
static inline void PCNML_MEDIAN(const Vec4f &kernal_x, const Vec4f &kernal_y,
                                const cv::Matx33d &camera,
                                const cv::Mat &input,
                                cv::Mat &output){

  output.create(input.rows,input.cols, CV_32FC3);//3 通道，非常make sense
  //x是竖着，y是横着的
  int COL = input.cols;
  cv::Vec3f* idx = (cv::Vec3f*)input.data + COL;
  cv::Vec3f* idx_m1 =(cv::Vec3f*)input.data;
  cv::Vec3f* idx_p1 = (cv::Vec3f*)input.data + COL + COL;
  cv::Vec3f* idx_o = (cv::Vec3f*)output.data + COL;
  cv::Vec3f* end_idx;

  int ONE_MINUS_COL = COL - 1;
  int v(2);
  const Vec4f ZERO(1e5);
  const Vec4f NEG_INF(-1e5);
  const float mul[5]={0,1,1/2.0,1/3.0,1/4.0};
#define UPDATE_IDX idx++, idx_m1++, idx_p1++, idx_o++
  for (; v!=input.rows; ++v){
    end_idx = idx + ONE_MINUS_COL;
    UPDATE_IDX;
    for (; idx != end_idx;UPDATE_IDX){
      Vec4f D = Vec4f(SURREND(2));
      float& nx = idx_o->operator()(0)=horizontal_add(kernal_x*camera(0,0) / D);
      float& ny = idx_o->operator()(1)=horizontal_add(kernal_y*camera(1,1) / D);

      if ((*((int*)&nx) & 0x7FFFFFFF) < 1e-7 &&(*((int*)&ny) & 0x7FFFFFFF) < 1e-7) {
        idx_o->operator()(2) = -1;
        continue;
      }
      Vec4f DX =  (idx->operator()(0) - Vec4f(SURREND(0))) * nx;
      Vec4f DY =  (idx->operator()(1) - Vec4f(SURREND(1))) * ny;

      float &x = idx->operator()(0);
      float &y = idx->operator()(1);
      Vec4f Z = D - idx->operator[](2);
      DX = (DX + DY) / Z;

      float *tmp=((float*)(&DX));
      int c=0;
      for (int i = 3; i >= 0; isnan(tmp[i])? i--: tmp[c++] = tmp[i--]);
      std::sort(tmp, tmp+c);
      if (c)  idx_o->operator()(2) = (c&1) ? tmp[c>>1] : (tmp[c>>1] + tmp[(c>>1)-1]) * 0.5;
      else idx_o->operator()(2) = 0;
      continue;
    }
    UPDATE_IDX;
  }
}

//直接操作，不用经过通道,调用理论最优参数。
static inline void PCNML2(const cv::Mat &range_image,
                          const cv::Matx33d camera,
                          const RIN_METHOD method,
                          cv::Mat* output) {
  const Vec8f kernel_x(-1, 0, 1, -2, 2, -1, 0, 1);
  const Vec8f kernel_y(-1, -2, -1, 0, 0, 1, 2, 1);
  const Vec4f kernel_x4(0, -1, 1, 0);
  const Vec4f kernel_y4(-1, 0, 0, 1);
  const Vec8f kernel_x48(0, 0, 0, -1, 1, 0, 0, 0);
  const Vec8f kernel_y48(0, -1, 0, 0, 0, 0, 1, 0);


  const Vec8f kernel_sobel_x(-1, 0, 1, -2, 2, -1, 0, 1);
  const Vec8f kernel_sobel_y(-1, -2, -1, 0, 0, 1, 2, 1);
  const Vec8f kernel_scharr_x(-3, 0, 3, -10, 10, -3, 0, 3);
  const Vec8f kernel_scharr_y(-3, -10, -3, 0, 0, 3, 10, 3);
  const Vec8f kernel_prewitt_x(-1, 0, 1, -1, 1, -1, 0, 1);
  const Vec8f kernel_prewitt_y(-1, -1, -1, 0, 0, 1, 1, 1);

  switch (method){
    case R_MEANS_8 :
      PCNML_MEAN(-kernel_x * camera(0,0), -kernel_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_FAST_8 :
      PCNML_MEDIAN_FAST(-kernel_x, -kernel_y, camera, range_image, *output);
      break;
    case R_MEDIAN_STABLE_8 :
      PCNML_MEDIAN_STABLE(-kernel_x, -kernel_y, camera, range_image, *output);
      break;
    case R_MEANS_4:
      PCNML_MEAN(-kernel_x4, -kernel_y4, camera, range_image, *output);
      break;
    case R_MEDIAN_4:
      PCNML_MEDIAN(-kernel_x4, -kernel_y4, camera, range_image, *output);
      break;
    case R_MEDIAN_FAST_4_8:
      PCNML_MEDIAN_FAST(-kernel_x48, -kernel_y48, camera, range_image, *output);
      break;
    case R_MEDIAN_STABLE_4_8:
      PCNML_MEDIAN_STABLE(-kernel_x48, -kernel_y48, camera, range_image, *output);
      break;
    case R_MEANS_4_8:
      PCNML_MEAN(-kernel_x48*camera(0,0), -kernel_y48*camera(1,1), range_image, *output);
      break;
    case R_MEANS_SOBEL:
      PCNML_MEAN(-kernel_sobel_x*camera(0,0), -kernel_sobel_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_SOBEL:
      PCNML_MEDIAN_STABLE(-kernel_sobel_x, -kernel_sobel_y, camera, range_image, *output);
      break;
    case R_MEANS_SCHARR:
      PCNML_MEAN(-kernel_scharr_x*camera(0,0), -kernel_scharr_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_SCHARR:
      PCNML_MEDIAN_STABLE(-kernel_scharr_x, -kernel_scharr_y, camera, range_image, *output);
      break;
    case R_MEANS_PREWITT:
      PCNML_MEAN(-kernel_prewitt_x*camera(0,0), -kernel_prewitt_y*camera(1,1), range_image, *output);
      break;
    case R_MEDIAN_PREWITT:
      PCNML_MEDIAN_STABLE(-kernel_prewitt_x, -kernel_prewitt_y, camera, range_image, *output);
      break;
    default:
      std::cerr<<"something wrong?" << std::endl;
      exit(-1);
  }

  //if (method == R_MEANS_8) {
  //  PCNML_MEAN(-kernal_x, -kernal_y, camera, range_image, *output);
  //}
//  if (method == R_MEDIAN_FAST_8){
//    PCNML_MEDIAN_FAST(-kernal_x, -kernal_y, camera, range_image, *output);
//  }
//  if (method == R_MEDIAN_STABLE_8){
//    PCNML_MEDIAN_STABLE(-kernal_x, -kernal_y, camera, range_image, *output);
//  }
  // if (method == R_MEANS_4){
  //   PCNML_MEAN(-kernal_x4, -kernal_y4, camera, range_image, *output);
  // }
//  if (method == R_MEDIAN_4){
//    PCNML_MEDIAN(-kernal_x4, -kernal_y4, camera, range_image, *output);
//  }
//  if (method == R_MEDIAN_FAST_4_8){
//    PCNML_MEDIAN_FAST(-kernal_x48, -kernal_y48, camera, range_image, *output);
//  }
//  if (method == R_MEDIAN_STABLE_4_8){
//    PCNML_MEDIAN_STABLE(-kernal_x48, -kernal_y48, camera, range_image, *output);
//  }
//  if (method == R_MEANS_4_8){
//    PCNML_MEAN(-kernal_x48, -kernal_y48, camera, range_image, *output);
//  }
//  //std::cout<<output.at<cv::Vec3f>(92, 380);
}


static inline cv::Mat PCNML(const cv::Mat &range_image,
                            const cv::Matx33d camera,
                            const RIN_METHOD method,
                            const bool need_normal){
  if (range_image.channels() != 3){
    exit(-1);
  }
  double st=clock();
  std::vector<cv::Mat> matpart(3);

  cv::split(range_image, matpart);
  //cv::Mat_<cv::Vec3f> r(range_image);

  matpart[0] = cv::Mat(cv::Size(range_image.cols,range_image.rows), CV_32F);
  matpart[1] = cv::Mat(cv::Size(range_image.cols,range_image.rows), CV_32F);
  matpart[2] = cv::Mat(cv::Size(range_image.cols,range_image.rows), CV_32F);

  cv::Mat_<float> x(matpart[0]);
  cv::Mat_<float> y(matpart[1]);
  cv::Mat_<float> z(matpart[2]);


  float* tt = (float*)range_image.data;
  float* idx_x = (float*)x.data;
  float* idx_y = (float*)y.data;
  float* idx_z = (float*)z.data;
  float* end;

  for (size_t i(0); i != range_image.rows; ++ i){
    end= tt + range_image.cols*3;
    for (; tt != end;) {
      *idx_x++ = *tt++;
      *idx_y++ = *tt++;
      *idx_z++ = *tt++;
    }
  }
  std::cout<<"Pretreatment time(split image):"<<  (clock()-st)/CLOCKS_PER_SEC << std::endl;
  //Imshow(x);
  //exit(-1);

  //printf("[[%lld %lld\n", matpart[0].datastart, matpart[0].dataend);
  //st=clock();
  //cv::split(range_image, matpart); ///////////
  //std::cout << (clock() - st)/CLOCKS_PER_SEC << std::endl;
  //printf("[[%lld %lld\n", matpart[0].datastart, matpart[0].dataend);


  return PCNML(matpart, camera, method, need_normal);
}










class RIN {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief 用一个range image初始化
   * @param range_image 输入的range image*/
  // RIN(const cv::Mat &range_image):range_image_(range_image){}
  //RIN(){}

  //void SetRangeImage(const cv::Mat &range_image){
  // // this->range_image_ = range_image;
  //}

  //void Init(const cv::Mat &range_image){
  //  SetRangeImage(range_image);
  //}

  static cv::Mat core(const cv::Mat &range_image){
    //auto &K = camera_parameters;
    cv::Mat KK(cv::Size(3,3),CV_32F);
    cv::Mat_<float> K(KK);
    K(1,1) = 1400;
    K(2,2) = 1400;

    if (range_image.channels() != 3){
      exit(-1);
    }

    std::vector<cv::Mat> matpart(3);
    cv::split(range_image, matpart);

    double st=clock(); //计算时间
    cv::Mat d = 1/matpart.at(2);
    cv::Mat Gv, Gu;

    //用sobel卷积先跑2个数据
    //cv::Mat sobel_y = (cv::Mat_<float>(3, 3)<< -1, -2, -1, 0, 0, 0, 1, 2, 1);
    //cv::Mat sobel_x = (cv::Mat_<float>(3, 3)<< -1, 0, 1, -2, 0, 2, -1, 0, 1);
    //cv::filter2D(d, Gv, CV_32F, -sobel_y);
    //cv::filter2D(d, Gu, CV_32F, -sobel_x);


    cv::Sobel(d, Gv, CV_32F, 0, 1, 3, -1400); //y方向
    cv::Sobel(d, Gu, CV_32F, 1, 0, 3, -1400); //x方向


    cv::Mat_<float> X(matpart.at(0));
    cv::Mat_<float> Y(matpart.at(1));
    cv::Mat_<float> Z(matpart.at(2));
    cv::Mat_<float> D(d);


    cv::Mat nx_t = Gu;/// * K(1,1);
    cv::Mat ny_t = Gv;// * K(2, 2);
    cv::Mat nz_t(d);
    cv::Mat_<float> nx(nx_t);
    cv::Mat_<float> ny(ny_t);
    cv::Mat_<float> nz(nz_t);
    {
      // Imshow(X);
      //  std::cout<<Gu.at<float>(246, 345)<<std::endl;
      //  std::cout<<Gv.at<float>(246, 345)<<std::endl;
      //  std::cout<<nx(246,345)<<std::endl;
      //  std::cout<<ny(246,345)<<std::endl;
      //  std::cout<<D(246,345)<<std::endl;
      //  std::cout<<X(246,345)<<" "<<X(247,346)<<std::endl;
      //  std::cout<<Y(246,345)<<" "<<Y(247,346)<<std::endl;
      //  std::cout<<Z(246,345)<<" "<<Z(247,346)<<std::endl;
      //  std::cout<< matpart[0].at<float>(247,346)<<std::endl;
      //  std::cout<<range_image.at<cv::Vec3f>(247,346)<< std::endl;
      //  std::cout<<"---------"<<std::endl;
      //  exit(-1);
    }
    float_t ret[8];
    float_t dx, dy, dz;
    std::cout <<"the tot time is : "<< (clock()-st)/CLOCKS_PER_SEC << std::endl;

    cv::Mat result(cv::Size(X.cols, X.rows), CV_32FC3);
    float NOT_USE[333];

    //显然按照经验v u比较快
    float tot=0;
    register_t u, v, i, xr(X.rows - 1), xc(X.cols -1);
    float t[8];
    //cv::Mat_<float>

    //Vec8f DX, DY, DZ;
    Vec8f ZERO(1e5);
    Vec8f NEG_INF(-1e5);
    Vec4i THIRTYTWO(31);

    register Vec8f DX;
    register Vec8f DY;
    register Vec8f DZ;
    register Vec8fb LT_ZERO ;
    register Vec8fb LG_NG_INF ;
    register Vec8fb BOOL_RESULT;//LT_ZERO & LG_NG_INF;
    register float sum;
    register float n;

    for (v = 1; v < xr; ++ v){
      for (u = 1; u < xc; ++ u){

        auto OUT_F=[](auto X){
          for (int i = 0; i < 8; ++ i){
            std::cout <<   ((float*)&X)[i] <<" ";
          }
          std::cout<<std::endl;
        };

        auto OUT_I=[](auto X){
          for (int i = 0; i < 8; ++ i){
            std::cout <<   ((unsigned int*)&X)[i] <<" ";
          }
          std::cout<<std::endl;
        };

        {
          //v=91;
          //u=379;
          // v=90;
          // u=378;
          //v=5;
          //u=5;
          // v=200;
          // u=200;
        }
//#define SLOW_VERSION
#ifdef SLOW_VERSION //纯普通实现
        {
          //debug
          const int_fast16_t du[]={0, 0, 1, 1, 1, -1, -1, -1};
          const int_fast16_t dv[]={1, -1, -1, 0, 1, -1, 0, 1};
          for (int i = 5; i <6; ++ i){
            //for (i = 4; i <5; ++ i){
            auto wu = u + du[i];
            auto wv = v + dv[i];
            dz = Z.at<float>(v, u) - Z.at<float>(wv, wu);
            dx = X.at<float>(v, u) - X.at<float>(wv, wu);
            dy = Y.at<float>(v, u) - Y.at<float>(wv, wu);
            ret[i] = -(nx_t.at<float>(v, u) * dx + ny_t.at<float>(v, u) * dy) / dz;
          }
          /*
          std::sort(ret, ret+8);
          result.at<cv::Vec3f>(v, u) = {nx_t.at<float>(v, u) ,
                                        ny_t.at<float>(v, u) ,
                                        (ret[3]+ret[4])/2};
          std::cout<< result.at<cv::Vec3f>(v, u) <<std::endl;
           */
          //continue;

        }

#endif
        //_mm_prefetch(&DX, _MM_HINT_NTA);
        if ((  (*(int*)&nx(v, u) & 0x7FFFFFFF) < 1e-7) &&  ((*(int*)&ny(v, u) & 0x7FFFFFFF) < 1e-7)){
          nz(v, u) = -1;
          continue;
        }


        DX = (X(v, u) - Vec8f(TT(X))) * nx(v, u);

        OUT_F(Vec8f(TT(X)));
        DY = (Y(v, u) - Vec8f(TT(Y))) * ny(v, u);
        DZ = Vec8f(TT(Z)) - Z(v, u);
        DX = (DX+DY) / DZ;

#ifdef MEAN_METHOD



        Vec8fb LT_ZERO = _mm256_cmp_ps(DX, ZERO, _CMP_LE_OS);



        Vec8fb LG_NG_INF = _mm256_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
        Vec8fb BOOL_RESULT = _mm256_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
        //  OUT_F(DX);
        //  OUT_I(LT_ZERO);
        //  OUT_I(LG_NG_INF);
        //  OUT_I(BOOL_RESULT);
        //  exit(-1);

        //T256(BOOL_RESULT);


        Vec8i x=_mm256_srli_epi32(_mm256_castps_si256(BOOL_RESULT), 31);
        float sum = horizontal_add((Vec8f)(_mm256_and_ps(T256(BOOL_RESULT), T256(DX))));
        //右移31位，然后统计求和就知道数字的数量，然后除一下。
        //float n = horizontal_add(to_float(BOOL_RESULT >> 31));

        auto n = horizontal_add(x);
        nz(v, u) = sum/n;
        //std::cout << nx(v, u) << " " <<ny(v, u)<<" "<<nz(v,u)<<std::endl;
        //exit(-1);
        continue;

        //Vec8i x=_mm256_sra_epi32(_mm256_castps_si256(BOOL_RESULT), THIRTYTWO);

        //horizontal_add((Vec8i)(BOOL_RESULT))

        //OUT_I(BOOL_RESULT);
        //OUT_I(THIRTYTWO);
        //OUT_I(x);
        //std::cout << sum << " "<<n << std::endl;


        //exit(-1);
        LT_ZERO = _mm256_cmp_ps(DX, ZERO, _CMP_LT_OS);
        LG_NG_INF = _mm256_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
        BOOL_RESULT = _mm256_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
        //T256(BOOL_RESULT);
        sum = horizontal_add((Vec8f)(_mm256_and_ps(T256(BOOL_RESULT), T256(DX))));
        //右移31位，然后统计求和就知道数字的数量，然后除一下。
        n = horizontal_add(to_float(BOOL_RESULT >> 31));
        //horizontal_add((Vec8i)(BOOL_RESULT))

        result.at<cv::Vec3f>(v, u) = {nx(v, u), ny(v, u), sum/n};
#endif

#ifdef MEDIAN_METHOD
        float *tmp=((float*)(&DX));
        //std::sort((float*)(&DX), ((float*)(&DX))+8);
        //for (int i = 0; i <5;++i)
        //  for (int j = i+1; j <8; ++ j){
        //    if (tmp[i]<tmp[j]){
        //      std::swap(tmp[i], tmp[j]);
        //    }
        //  }
        OUT_F(DX);

        {
          //手动中值?
          if (tmp[1] <  tmp[0])  std::swap(tmp[0], tmp[1]);
          if (tmp[2] <  tmp[1])  std::swap(tmp[1], tmp[2]);
          // if (tmp[3] <  tmp[2])  std::swap(tmp[2], tmp[3]); //可能要放出来
          // if (tmp[4] <  tmp[3])  std::swap(tmp[3], tmp[4]);
          if (tmp[5] <  tmp[4])  std::swap(tmp[4], tmp[5]);
          //if (tmp[6] <  tmp[5])  std::swap(tmp[5], tmp[6]);
          if (tmp[7] <  tmp[6])  std::swap(tmp[6], tmp[7]);
          //max 3     max 7

          if (tmp[1] <  tmp[0])  std::swap(tmp[0], tmp[1]);
          //if (tmp[2] <  tmp[1])  std::swap(tmp[1], tmp[2]);
          if (tmp[3] <  tmp[2])  std::swap(tmp[2], tmp[3]);
          if (tmp[4] <  tmp[3])  std::swap(tmp[3], tmp[4]);
          //if (tmp[5] <  tmp[4])  std::swap(tmp[4], tmp[5]);
          if (tmp[6] <  tmp[5])  std::swap(tmp[5], tmp[6]);

          if (tmp[1] <  tmp[0])  std::swap(tmp[0], tmp[1]);
          if (tmp[2] <  tmp[1])  std::swap(tmp[1], tmp[2]);
          if (tmp[3] <  tmp[2])  std::swap(tmp[2], tmp[3]);
          //if (tmp[4] <  tmp[3])  std::swap(tmp[3], tmp[4]);
          if (tmp[5] <  tmp[4])  std::swap(tmp[4], tmp[5]);

          if (tmp[1] <  tmp[0])  std::swap(tmp[0], tmp[1]);
          if (tmp[2] <  tmp[1])  std::swap(tmp[1], tmp[2]);
          if (tmp[3] <  tmp[2])  std::swap(tmp[2], tmp[3]);
          if (tmp[4] <  tmp[3])  std::swap(tmp[3], tmp[4]);

          //if (tmp[1] <  tmp[0])  std::swap(tmp[0], tmp[1]);
          // if (tmp[2] <  tmp[1])  std::swap(tmp[1], tmp[2]);
          if (tmp[3] <  tmp[2])  std::swap(tmp[2], tmp[3]);
        }
        //std::sort((float*)(&DX), ((float*)(&DX))+8);
        nz_t.at<float>(v, u) = (tmp[3] + tmp[4]) / 2;
        std::cout<< nx(v, u) << " "<<ny(v, u) << nz(v, u)<<std::endl;
        exit(-1);
        continue;
#endif
      }
    }
    std::cout <<"the tot time is :"<< (clock()-st)/CLOCKS_PER_SEC << std::endl;

    //归一化
    for (int u = 1; u < result.cols - 1; ++ u)
      for (int v = 1; v < result.rows - 1; ++ v){
        cv::Vec3f x = {nx(v, u), ny(v, u), nz(v, u)};
        // std::cout<<u<<" "<<v<<std::endl;
        float d = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));
        x(0) /= d;
        x(1) /= d;
        x(2) /= d;
        if (x(2)>0){
          x(0) = - x(0);
          x(1) = - x(1);
          x(2) = - x(2);
        }
        result.at<cv::Vec3f>(v, u) = x;
      }
    return result;
  }
};

#endif //PCNML_RIN_H_


/*
    for (u = 1; u < xc; ++ u){
     if ((  (*(int*)&nx(v, u) & 0x7FFFFFFF) < 1e-7) &&  ((*(int*)&ny(v, u) & 0x7FFFFFFF) < 1e-7)){
        nz(v, u) = -1;
        continue;
      }

      DX = (X(v, u) - Vec8f(TT(X))) * nx(v, u);
      DY = (Y(v, u) - Vec8f(TT(Y))) * ny(v, u);
      DZ = Vec8f(TT(Z)) - Z(v, u);
      DX = (DX+DY) / DZ;

      Vec8fb LT_ZERO = _mm256_cmp_ps(DX, ZERO, _CMP_LE_OS);
      Vec8fb LG_NG_INF = _mm256_cmp_ps(NEG_INF, DX, _CMP_LT_OS);
      Vec8fb BOOL_RESULT = _mm256_and_ps(LT_ZERO, LG_NG_INF);//LT_ZERO & LG_NG_INF;
      Vec8i x=_mm256_srli_epi32(_mm256_castps_si256(BOOL_RESULT), 31);
      float sum = horizontal_add((Vec8f)(_mm256_and_ps(T256(BOOL_RESULT), T256(DX))));
      size_t n = horizontal_add(x);
      nz(v, u) = sum*mul[n];
    }
  }
  */

