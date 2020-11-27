//
// Created by bohuan on 2019/10/18.
//

#ifndef PCNML_UTILS_VISUALIZATION_H_
#define PCNML_UTILS_VISUALIZATION_H_
#include <opencv2/opencv.hpp>

static void Imshow(cv::Mat i){
  cv::Mat t = i.clone();
  cv::Mat q = i;
  cv::Mat_<float> c = q;
  float maxx=-1e30, minn=1e30;
  for (auto it : c){
    if (it > maxx){
      maxx = it;
    }
    if (it <minn){
      minn = it;
    }
  }
  auto d = maxx-minn;
  for (auto &it : c){
    it = (it - minn) / d;
  }
  std::cout<<maxx<<" -- "<<minn<<std::endl;
  cv::imshow("a",i);
  cv::waitKey(-1);
  i = t.clone();

}

//class visualization {

//};

#endif //PCNML_UTILS_VISUALIZATION_H_
