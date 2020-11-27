//
// Created by bohuan on 2020/11/27.
//
//已经忘了都是干嘛的了。完了。

#include <cstdlib>
#include <iostream>
#include <eigen3/Eigen/Core>
#include "pcnml/RIN.h"

#include "pcnml/IO/LoadHPP.h"
#include "pcnml/utils/est_result.h" //用来评估结果


#include "cvrgbd/rgbd.hpp"

#include "pcnml/ThirdAlgorithm/ThirdAlgorithm.h"

/**
  * @brief 读取深度图像,范围是[0,1]范围内的
  * 按照数据，格式应该是bin格式。
  * */
cv::Mat LoadDepthImage(const std::string &path, const size_t width = 640,
                       const size_t height = 480){
  const int buffer_size = sizeof(float) * height * width;
  //char *buffer = new char[buffer_size];

  cv::Mat mat(cv::Size(width, height), CV_32FC1);

  // open filestream && read buffer
  std::ifstream fs_bin_(path, std::ios::binary);
  fs_bin_.read(reinterpret_cast<char*>(mat.data), buffer_size);
  fs_bin_.close();
  return mat;
}

int main(){

  //这行是干嘛的，我也忘了
  cv::Matx33d camera(0,0,0,0,0,0,0,0,1);

  cv::Mat range_image;
  cv::Mat result;
  cv::Mat save;
  cv::Mat output;


  //参数地址
  int n; //文件夹里图片的个数,忽略掉，这个没啥用.用来读王恒力的param
  std::string param = "/home/zpmc/bohuan/code/pcnml/data/android/params.txt";
  FILE *f = fopen(param.c_str(), "r");
  fscanf(f, "%lf %lf %lf %lf %d", &camera(0,0),
         &camera(1,1), &camera(0, 2), &camera(1,2), &n);
  camera(0,2)--;  camera(1,2)--;
  std::cout<<"相机K:"<< camera << std::endl;


  //读入bin的深度图片，并转化为range_image. 印象中有bgr和rgb的问题，但是我忘了。
  auto lppp = new LoadHPP();
  LoadHPP &lpp = *lppp;
  auto depth_image = lpp.LoadDepthImage("/home/zpmc/bohuan/code/pcnml/data/android/depth/000001.bin", 640, 480);
  cv::Mat_<float> s(depth_image);
  for (auto &it : s){
    if (fabs(it) < 1e-7){
      it = 1e10;
    }
  }
  cv::rgbd::depthTo3d(depth_image, camera, range_image);
  //分离图像通道
  std::vector<cv::Mat> matpart(3);
  cv::split(range_image, matpart);

  result.create(matpart[0].rows, matpart[0].cols, CV_32FC3);//3 通道，非常make sense

  /*****************************************/
  PCNML_ICIP(range_image, camera, result); //这行代码是调用的函数。
  /*****************************************/

  auto makeoutput=[&](){//对result数组赋值成符合output格式.(因为bgr格式，所以x,z交换位置) 这个注释是去年写的，我已经不记得了。
    output.create(result.rows, result.cols, CV_16UC3);
    for (int i = 0; i < result.rows; ++ i){
      for (int j = 0; j < result.cols; ++ j){
        result.at<cv::Vec3f>(i, j) = result.at<cv::Vec3f>(i, j) / cv::norm(result.at<cv::Vec3f>(i, j));
        if (result.at<cv::Vec3f>(i, j)[2] < 0) {
          result.at<cv::Vec3f>(i, j) = -result.at<cv::Vec3f>(i, j);
        }
        output.at<cv::Vec3w>(i, j)[2] = (result.at<cv::Vec3f>(i, j)[0]+1)*(65535/2.0);
        output.at<cv::Vec3w>(i, j)[1] = (result.at<cv::Vec3f>(i, j)[1]+1)*(65535/2.0);
        output.at<cv::Vec3w>(i, j)[0] = (result.at<cv::Vec3f>(i, j)[2]+1)*(65535/2.0);
      }
    }
  };
  makeoutput();
  cv::imshow("result", output);
  cv::waitKey(-1);





}

