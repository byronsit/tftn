//
// Created by bohuan on 2019/10/24.
//

#ifndef PCNML_INCLUDE_PCNML_UTILS_EST_RESULT_H_
#define PCNML_INCLUDE_PCNML_UTILS_EST_RESULT_H_

#include <opencv2/opencv.hpp>
#include "pcnml/utils/utils.hpp"

static float norm(cv::Vec3f x){
  return sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));
}
//误差多少角度，角度角度
static float Compare2NormalDegree(cv::Vec3f a, cv::Vec3f b){
  std::swap(a(0), a(2));
  double d = sqrt((double)b(0)*b(0)  + (double)b(1)*b(1) + (double)b(2)*b(2));
  b = b / d;

  d=sqrt(a(0)*a(0) + a(1)*a(1)+a(2)*a(2));
  a = a / d;

  double tt =fabs(((double)a(0)*b(0) + (double)a(1)*b(1) + (double)a(2)*b(2)));
  if (tt>1) {
    return 0;
    //不应当啊!
    std::cout<<"wa?"<<std::endl;
    std::cout<<a<<" "<<b<<tt<<std::endl;
    exit(-1);
  }
  double ret = acos(tt) / M_PI * 180;
  if (ret > 90) ret = 180 - ret;
 return ret;

}

static float ddd(const Mask_t &t){

}

static float Compare2image(const cv::Mat_<cv::Vec3f> &a,
                           const cv::Mat_<cv::Vec3f> &b,const Mask_t &mask){
  float tot=0;
  int cnt =0;
  int nan_number(0);
  float max=-1e5;
  float min=1e5;
  for (int x = 1; x < a.rows - 1; ++ x)
    for (int y = 1; y < a.cols - 1; ++ y) {
      if (mask[x][y]) {
        float t = Compare2NormalDegree(a(x, y), b(x, y));
        if (isnan(t)) {
          ++nan_number;
          //std::cout<<x+1<<" "<<y+1<<std::endl;
          continue;
        }
        if (t>20){
        //  std::cout<<x<<" "<<y<<std::endl;
        }
        tot+=t;
        max=std::max(t, max);
        min=std::min(t, min);
        ++cnt;
      }
    }
  std::cout <<"-------------------------\n";
  std::cout <<"the NaN number is: " << nan_number << std::endl;
  std::cout << "vaild number is:  " << cnt << std::endl;
  std::cout << "error :           " << tot/cnt << " degree"<<std::endl;
  std::cout <<"min errror:        " << min << " degree"<<std::endl;

  std::cout <<"max errror:        " << max << " degree"<<std::endl;
  return tot/cnt;
}


#endif //PCNML_INCLUDE_PCNML_UTILS_EST_RESULT_H_
