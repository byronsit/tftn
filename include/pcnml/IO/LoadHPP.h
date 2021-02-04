//
// Created by bohuan on 2019/10/18.
//

#ifndef PCNML_IO_LOADHPP_H_
#define PCNML_IO_LOADHPP_H_

#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <pcnml/utils/utils.hpp>

/**
 * @brief 读取黄皮皮的数据集*/
class LoadHPP {
 public:
  /**
   * @brief 读取HPP的gt,数据是16bit的RGB图像
   * 要读取的数据应该是*.bin格式的*/
   Mask_t& Mask(){
     return this->mask_;
   }

  //std::vector<std::vector<cv::Vec3f>> LoadGroundTruthInt16(const std::string path){

  cv::Mat LoadGroundTruthInt16(const std::string path){
    //cv::Mat input = cv::imread(path, cv::IMWRITE_PAM_FORMAT_RGB);

    cv::Mat input = cv::imread(path, -1);
    //std::vector<std::vector<cv::Vec3f>> normal_ground_truth;

    auto height = input.rows;
    auto width = input.cols;


    //先长再宽
    //std::cout<<width<<" "<<height << std::endl;
    cv::Mat output(cv::Size(width,height), CV_32FC3, cv::Scalar::all(0));

    //normal_ground_truth.resize(width);
    //for (auto &it : normal_ground_truth) it.resize(length);
    //cv::Mat_<uint16_t > in(input);
    //cv::Mat_<float> ou(output);

    for (int x = 0; x < output.rows; ++ x){
      for (int y = 0; y < output.cols; ++ y){
//        {
//          y=246;
//          x=345;
//          cv::Vec3w p = input.at<cv::Vec3w>(x, y);
//          cv::Vec3f q = cv::Vec3f({(float)p(0),(float)p(1) , (float)p(2) }) ;
//          //if (p(0) == 65535)continue;
//          q = q  / 65535.0 * 2;
//          // std::cout <<"@"<< q << std::endl;
//          q(0) -= 1;  q(1) -= 1;  q(2) -= 1;
//          //output.at<cv::Vec3f>(x, y) = q;
//          std::cout << p << std::endl;
//          std::cout << q << std::endl;
//          exit(-1);
//        }
        cv::Vec3w p = input.at<cv::Vec3w>(x, y);
        cv::Vec3f q = cv::Vec3f({(float)p(0),(float)p(1) , (float)p(2) }) ;
        //if (p(0) == 65535)continue;
        q = q  / 65535.0 * 2;
       // std::cout <<"@"<< q << std::endl;
        ///(65535/2.0);
        q(0) -= 1;  q(1) -= 1;  q(2) -= 1;
        std::swap(q(0), q(2));
        if (p(0) != 65535) {
         // std::cout<<p(0)<<std::endl;
          output.at<cv::Vec3f>(x, y) = q;
          mask_.Set(x, y);
        }
      }
    }

 //   cv::imshow("x", input);
//    cv::waitKey(-1);
    return output;//normal_ground_truth;
  }

  /**
   * @brief 读取深度图像,范围是[0,1]范围内的
   * 按照hpp的数据，格式应该是bin格式。
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

    //cv::Mat mat(cv::Size(width, height), CV_32FC1);
    //memcpy(mat.data, buffer, buffer_size);
    //delete [] buffer;
    //cv::imshow("x",mat);
    //cv::waitKey(-1);
    return mat;
  }

  //读取深度图，格式是png图片格式
  cv::Mat LoadDepthImagePng(const std::string &path, const size_t width = 640,
                            const size_t height = 480){
    //cv::Mat mat(cv::Size(width, height), CV_32FC1);
    cv::Mat mat = cv::imread(path, 0);
    std::cout<<mat.size()<<std::endl;
    std::cout<<mat.channels()<<std::endl;
    std::cout<<mat.type() << std::endl;
    cv::Mat mat2(mat.size(), CV_32FC1);
    for (int i = 0; i < mat2.size().width * mat2.size().height; ++ i){
      mat2.at<float>(i) = mat.at<uint8_t>(i)/255.0;
    }

   // cv::imshow("a",mat2);
   // cv::waitKey(-1);
   // exit(-1);
    return mat2;
  }

  /**
   * @brief 设置相机内参
   * 就驶入正常的一个内参矩阵就行了*/
  void SetCameraInternalParamteres(
      const Eigen::Matrix3d &camera_parameters){
    this->camera_parameters_ = camera_parameters;
  }

 private:

  Eigen::Matrix3d camera_parameters_;
  Mask_t mask_;
  //size_t length_, width_;//长宽
  //std::array<std::array<bool, 1<<10>, 1<<10> mask_;
  //std::vector<std::vector<cv::Vec3f>> normal_ground_truth_;




};

#endif //PCNML_IO_LOADHPP_H_
