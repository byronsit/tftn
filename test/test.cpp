//
// Created by bohuan on 2019/10/18.
//

#include <cstdlib>
#include <iostream>
#include <eigen3/Eigen/Core>
#include "pcnml/RIN.h"

#include "pcnml/IO/LoadHPP.h"
#include "pcnml/utils/est_result.h" //用来评估结果


#include "cvrgbd/rgbd.hpp"

#include "pcnml/ThirdAlgorithm/ThirdAlgorithm.h"


int start_from = 1; //下标从1开始

//X::operator bool (){}



/**
 * @brief 给定 depth图像，和相机内参以及z_factor，
 * 求出每个像素真实的数值*/
cv::Mat CalActualValue(cv::Mat &depth, Eigen::Matrix3d &camera_parameters,
                    const double z_factor = 600){
  cv::Mat output(cv::Size(depth.cols, depth.rows), CV_32FC3);
  auto fx = camera_parameters(0,0);
  auto fy = camera_parameters(1,1);;
  auto cx = camera_parameters(0,2) - 1;
  auto cy = camera_parameters(1,2) - 1;
  for (int u = 0; u< depth.cols; ++ u){ //x
    for (int v = 0; v < depth.rows; ++ v){ //y
    //  {
    //    v = 247;
    //    u = 346;
    //    auto d = depth.at<float>(v, u);
    //    double z = d * z_factor;
    //    double x = (u - cx) *z / fx;
    //    double y = (v - cy) *z / fy;
    //    output.at<cv::Vec3f>(v, u) = cv::Vec3f(x, y, z);
    //    std::cout<< cv::Vec3f(x, y, z) << std::endl;
    //    exit(-1);
    //  }
      auto d = depth.at<float>(v, u);
      double z = d * z_factor;
      double x = (u - cx) *z / fx;
      double y = (v - cy) *z / fy;
      output.at<cv::Vec3f>(v, u) = cv::Vec3f(x, y, z);
    }
  }
  return output;
}

RIN root;

LoadHPP *lppp;

void DEV(){
  std::string path = "/home/zpmc/bohuan/for_paper/aa/";
  std::string param = path + "params.txt";
  cv::Matx33d camera(0,0,0,0,0,0,0,0,1);
  cv::Mat range_image;
  cv::Mat result;
  cv::Mat save;
  cv::Mat output;
  FILE *f = fopen(param.c_str(), "r");
  // std::cout << param << std::endl;
  int n; //the number of data.
  fscanf(f, "%lf %lf %lf %lf %d", &camera(0,0),
         &camera(1,1), &camera(0, 2), &camera(1,2), &n);
  camera(0,2)--;  camera(1,2)--;
  fclose(f);
  std::vector<float> time;
  //std::vector<float>


  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(100);

  for (int i = 1; i <= 2;  i += 1){
    std::cout<<i<<std::endl;
    std::stringstream file_name;
    file_name<< std::setw(6) << std::setfill('0') << i ;
    lppp = new LoadHPP();
    LoadHPP &lpp = *lppp;
    //读入bin文件
    //auto depth_image = lpp.LoadDepthImage((path + "depth/" + file_name.str()+".bin").c_str(), 640, 480);

    //读入png文件
    auto depth_image = lpp.LoadDepthImagePng((path + "depth/" + file_name.str()+".png").c_str(), 640, 480);


    cv::Mat_<float> s(depth_image);
    for (auto &it : s){
      if (fabs(it) < 1e-7){
        it = 1e10;
      }
    }
    cv::rgbd::depthTo3d(depth_image, camera, range_image);

    double start_time;
    auto makeoutput=[&](){//对result数组赋值成符合output格式.(因为bgr格式，所以x,z交换位置)
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
    std::vector<cv::Mat> matpart(3);
    cv::split(range_image, matpart);
    //Imshow(matpart[0]);

    //Imshow(matpart[1]);
    //Imshow(matpart[2]);
    //cv::imshow("aaa", range_image);
    //cv::waitKey(-1);
    //exit(-1);





    result.create(matpart[0].rows, matpart[0].cols, CV_32FC3);//3 通道，非常make sense


    double st= clock();

    PCNML_FAST(matpart, camera, &result);
    //PCNML_ICIP(range_image, camera, result);
    //PCNML2(range_image, camera, R_MEDIAN_FAST_4_8, &result);
    //PCNML_FAST()

    std::cout << (clock() -st ) /CLOCKS_PER_SEC << std::endl;

    makeoutput();
    cv::imshow("result", output);
    cv::waitKey(-1);

    cv::imwrite(path+"result/" + "TEST" + "/" + file_name.str() + ".png" , output, compression_params);




//#define MAKE_CLOCK(METHOD)\
//    start_time = clock();\
//    PCNML2(range_image, camera, METHOD, &result);\
//    time.push_back((clock() - start_time) / CLOCKS_PER_SEC);\
//    makeoutput();\
//    cv::imshow("result", output);\
//    cv::waitKey(-1);
//    MAKE_CLOCK(R_MEANS_4_8);
//
//#undef MAKE_CLOCK
    delete lppp;
  } //end of: for (int i = 1; i <= n; ++ i){

}


//我要获得最好的kernal!
void GET_BEST_KERNAL(){


  LoadHPP lpp;
  std::stringstream sss,ss;
  ss << "/media/bohuan/data/dataset/pcnml/hard/batman/depth/" << std::setw(6) << std::setfill('0')
     << 1 << ".bin";
  auto depth_image = lpp.LoadDepthImage(ss.str());
  sss << "/media/bohuan/data/dataset/pcnml/hard/batman/normal/" << std::setw(6) << std::setfill('0')
      << 1 << ".png";
  cv::Mat gt = lpp.LoadGroundTruthInt16(sss.str());

  cv::Matx33d camera; //相机参数
  camera << 1400, 0, 350-1, 0, 1400, 200-1, 0, 0, 1;
  cv::Mat range_image;// = CalActualValue(depth_image,camera_parameters ,600);
  cv::rgbd::depthTo3d(depth_image*600, camera, range_image);
  cv::Mat result;


  Vec8f kx;
  Vec8f ky;
  double min_error = 1e10;
  for (float x1=-5; x1<=5;  x1 += 1)
    for (float x2 = -5; x2 <= 5; x2 += 1)
      for (float x3 = -5; x3 <= 5; x3 += 1)
        for (float x4 = -5; x4 <= 5; x4 += 1){
          //   {
          x1=0;
          x2=0;
          x3=-4.5;
          x4=0;
          //   }
          kx= Vec8f(x1, x2, -x4, x3, -x3, x4, -x2, -x1);
          //ky= Vec8f(x1, x3, x4, x2, -x2, -x4, -x3, -x1);
          ky= Vec8f(x4,x3,x1, -x2, x2,-x1,-x3,-x4);

          //result = PCNML2(range_image, camera, R_MEANS);
          PCNML_MEAN(-kx*camera(0,0), -ky*camera(1,1), range_image, result);

          double q=Compare2image(result, gt, lpp.Mask());
          if (q<min_error){
            std::cout << x1 <<" "<<x2<<" "<<x3<< " "<<x4 <<" ,and the error is:"<<q<< std::endl;
            min_error = q;
          }
        }


}

#include <cstdio>
//the end of path need the char '/'
void TEST(std::string path, int K){
  system(("mkdir " + path+"result").c_str());
  system(("mkdir " + path+"result/ANGLEWEIGHTED").c_str());
  system(("mkdir " + path+"result/AREAWEIGHTED").c_str());
  system(("mkdir " + path+"result/FLAS").c_str());
  system(("mkdir " + path+"result/LINEMOD").c_str());
  system(("mkdir " + path+"result/PLANEPCA").c_str());
  system(("mkdir " + path+"result/PLANESVD").c_str());
  system(("mkdir " + path+"result/SRI").c_str());
  system(("mkdir " + path+"result/VECTORSVD").c_str());

  std::string param = path + "params.txt";
  cv::Matx33d camera(0,0,0,0,0,0,0,0,1);
  cv::Mat range_image;
  cv::Mat result;
  cv::Mat save;
  cv::Mat output;
  FILE *f = fopen(param.c_str(), "r");
  // std::cout << param << std::endl;
  int n; //the number of data.
  fscanf(f, "%lf %lf %lf %lf %d", &camera(0,0),
         &camera(1,1), &camera(0, 2), &camera(1,2), &n);
  camera(0,2)--;  camera(1,2)--;
  fclose(f);
  std::vector<float> time_PLANESVD(0);
  std::vector<float> time_PLANEPCA(0);
  std::vector<float> time_VECTORSVD(0);
  std::vector<float> time_QUADSVD(0);
  std::vector<float> time_QUADTRANSSVD(0);
  std::vector<float> time_AREAWEIGHTED(0);
  std::vector<float> time_ANGLEWEIGHTED(0);
  std::vector<float> time_R_MEANS_8(0);
  std::vector<float> time_R_MEDIAN_FAST_8(0);
  std::vector<float> time_R_MEDIAN_STABLE_8(0);
  std::vector<float> time_R_MEANS_4(0);
  std::vector<float> time_R_MEDIAN_4(0);
  std::vector<float> time_R_MEDIAN_FAST_4_8(0);
  std::vector<float> time_R_MEDIAN_STABLE_4_8(0);
  std::vector<float> time_R_MEANS_4_8(0);
  std::vector<float> time_R_MEANS_SOBEL(0);
  std::vector<float> time_R_MEDIAN_SOBEL(0);
  std::vector<float> time_R_MEANS_SCHARR(0);
  std::vector<float> time_R_MEDIAN_SCHARR(0);
  std::vector<float> time_R_MEANS_PREWITT(0);
  std::vector<float> time_R_MEDIAN_PREWITT(0);
  std::vector<float> time_FLAS(0);
  std::vector<float> time_LINEMOD(0);
  std::vector<float> time_SRI(0);
  //std::vector<float>


  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(100);
  cv::rgbd::RgbdNormals FLAS(480, 640, CV_32F, camera, 3,
                             cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS);
  FLAS.initialize();
  cv::rgbd::RgbdNormals SRI(480, 640, CV_32F, camera, 3,
                            cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_SRI);
  SRI.initialize();
  cv::rgbd::RgbdNormals LINEMOD(480, 640, CV_32F, camera, 3,
                                cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
  LINEMOD.initialize();


  for (int i = start_from; i <= n;  i += 1){
    std::cout<<i<<std::endl;
    std::stringstream file_name;
//std::stringstream snormal;
    file_name<< std::setw(6) << std::setfill('0') << i ;
    lppp = new LoadHPP();
    LoadHPP &lpp = *lppp;
    auto depth_image = lpp.LoadDepthImage((path + "depth/" + file_name.str()+".bin").c_str(), 640, 480);
    cv::Mat_<float> s(depth_image);
    for (auto &it : s){
      if (fabs(it) < 1e-7){
        it = 1e10;
      }
    }
//std::cout<<(path + "depth/" + file_name.str()).c_str() << std::endl;
    cv::rgbd::depthTo3d(depth_image, camera, range_image);
    double start_time;
    auto makeoutput=[&](){//对result数组赋值成符合output格式.(因为bgr格式，所以x,z交换位置)
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

#define MAKE_CLOCK(METHOD)\
    start_time = clock();\
    GetNormal(range_image, K, &result, METHOD);\
    time_##METHOD.push_back((clock() - start_time) / CLOCKS_PER_SEC);\
    makeoutput();\
    cv::imwrite(path+"result/" + #METHOD + "/" + file_name.str() + ".png" , output, compression_params);
    MAKE_CLOCK(PLANESVD);
    MAKE_CLOCK(VECTORSVD);
    MAKE_CLOCK(AREAWEIGHTED);
    MAKE_CLOCK(ANGLEWEIGHTED);
    MAKE_CLOCK(PLANEPCA);
//MAKE_CLOCK(QUADSVD); 废弃
//MAKE_CLOCK(QUADTRANSSVD);  废弃
#undef MAKE_CLOCK

//opencv自带的函数
#define MAKE_CLOCK(METHOD)\
    start_time = clock();\
    METHOD(range_image, result);\
    time_##METHOD.push_back((clock() - start_time) / CLOCKS_PER_SEC);\
    makeoutput();\
    cv::imwrite(path+"result/" + #METHOD + "/" + file_name.str() + ".png" , output, compression_params);
    MAKE_CLOCK(FLAS);
    MAKE_CLOCK(LINEMOD);
    MAKE_CLOCK(SRI);
#undef MAKE_CLOCK

#define MAKE_CLOCK(METHOD)\
    start_time = clock();\
    PCNML2(range_image, camera, METHOD, &result);\
    time_##METHOD.push_back((clock() - start_time) / CLOCKS_PER_SEC);\
    makeoutput();\
    cv::imwrite(path+"result/" + #METHOD + "/" + file_name.str() + ".png" , output, compression_params);
    //MAKE_CLOCK(R_MEANS_4_8);
    //MAKE_CLOCK(R_MEANS_PREWITT);
    //MAKE_CLOCK(R_MEANS_SCHARR);
    //MAKE_CLOCK(R_MEANS_SOBEL);

    //MAKE_CLOCK(R_MEDIAN_STABLE_4_8);
    //MAKE_CLOCK(R_MEDIAN_SOBEL)
    //MAKE_CLOCK(R_MEDIAN_SCHARR);
    //MAKE_CLOCK(R_MEDIAN_PREWITT);
//MAKE_CLOCK(R_MEANS_8);
//MAKE_CLOCK(R_MEDIAN_STABLE_8);
//MAKE_CLOCK(R_MEDIAN_FAST_8);
//MAKE_CLOCK(R_MEANS_4);
//MAKE_CLOCK(R_MEDIAN_4);
//MAKE_CLOCK(R_MEDIAN_FAST_4_8);
#undef MAKE_CLOCK
    delete lppp;
  } //end of: for (int i = 1; i <= n; ++ i){

#define SAVE_RESULT(METHOD) \
  f = fopen((path+"result/"+ #METHOD + "_result.txt").c_str(), "w");\
  for (auto it : time_##METHOD){\
    fprintf(f, "%lf\n", it);\
  }\
  fclose(f);

//SAVE_RESULT(QUADSVD);
//SAVE_RESULT(QUADTRANSSVD);
// fprintf(f, (std::string(#METHOD)+": %lf\n").c_str(), time_##METHOD);

  SAVE_RESULT(PLANESVD);
  SAVE_RESULT(PLANEPCA);
  SAVE_RESULT(VECTORSVD);
  SAVE_RESULT(AREAWEIGHTED);
  SAVE_RESULT(ANGLEWEIGHTED);

  //SAVE_RESULT(R_MEANS_4_8);
  //SAVE_RESULT(R_MEANS_PREWITT);
  //SAVE_RESULT(R_MEANS_SOBEL);
  //SAVE_RESULT(R_MEANS_SCHARR);

  //SAVE_RESULT(R_MEDIAN_STABLE_4_8);
  //SAVE_RESULT(R_MEDIAN_PREWITT);
  //SAVE_RESULT(R_MEDIAN_SOBEL);
  //SAVE_RESULT(R_MEDIAN_SCHARR);

  SAVE_RESULT(SRI);
  SAVE_RESULT(LINEMOD);
  SAVE_RESULT(FLAS);
//下面的算法都是我们的
//SAVE_RESULT(R_MEANS_4);
//SAVE_RESULT(R_MEDIAN_FAST_8);
//SAVE_RESULT(R_MEDIAN_4);
//SAVE_RESULT(R_MEANS_4_8);
//SAVE_RESULT(R_MEANS_8);
//SAVE_RESULT(R_MEDIAN_STABLE_8);

#undef SAVE_RESULT

  fclose(f);
}

int main(int argc, char* argv[]){

  DEV();
  return 0;
  //TEST();
  //start_from = atoi(argv[2]);
  //TEST(argv[1], 10);
  TEST("/home/zpmc/bohuan/for_paper/Easy/boat/", 10); //记得路径结尾有 /
  //TEST("/media/bohuan/data/dataset/pcnml/easy/torusknot/", 10);

  //TEST("/media/bohuan/data/dataset/pcnml/easy/debug/torusknot/", 10);
  //TEST("/media/bohuan/data/dataset/pcnml/easy/android/", 10);
  return 0;
  std::string input_path="/home/bohuan/code/glzoo/torusknot/depth/";
  double tot=0;
  double t_err=0;
  double tot_cv=0;
  int cnt=1800; //数据数量
  std::stringstream sss,ss;
  ss << "/media/bohuan/data/dataset/pcnml/hard/batman/depth/" << std::setw(6) << std::setfill('0')
     << 1 << ".bin";
  lppp = new LoadHPP();
  LoadHPP &lpp = *lppp;
  auto depth_image = lpp.LoadDepthImage(ss.str());
  cv::Matx33d camera; //相机参数
  //camera << 1400, 0, 350-1, 0, 1400, 200-1, 0, 0, 1;
  camera << 1400, 0, 300-1, 0, 1380, 420-1, 0, 0, 1;
  cv::rgbd::RgbdNormals flas(depth_image.rows,depth_image.cols, CV_32F,camera,3,cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS);
  flas.initialize();

  for (int i = 1; i <= cnt; ++ i){
    std::stringstream sss,ss;
    ss << "/media/bohuan/data/dataset/pcnml/hard/batman/depth/" << std::setw(6) << std::setfill('0')
       << i << ".bin";
    lppp = new LoadHPP();
    LoadHPP &lpp = *lppp;
    auto depth_image = lpp.LoadDepthImage(ss.str());
    //std::cout<<depth_image.rows << " " << depth_image.cols << std::endl;

    sss << "/media/bohuan/data/dataset/pcnml/hard/batman/normal/" << std::setw(6) << std::setfill('0')
        << i << ".png";
    cv::Mat gt = lpp.LoadGroundTruthInt16(sss.str());
    cv::Mat range_image;// = CalActualValue(depth_image,camera_parameters ,600);
    cv::rgbd::depthTo3d(depth_image*600, camera, range_image);
    cv::Mat result;
  }

  return 0;
}

