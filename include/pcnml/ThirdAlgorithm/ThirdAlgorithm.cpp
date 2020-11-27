//
// Created by bohuan on 2019/11/7.
//

#include "pcnml/ThirdAlgorithm/ThirdAlgorithm.h"

#ifdef USE_KNN
/**
 * @brief 输入range_image，输出一个数组
 * @param output output[x][y][p]表示图像第x,y位置上，第k近的点是什么
 * @param K 搜索k邻近点(包含自己)*/
static void GetNearPoint(const cv::Mat &range_image,
                                const int K,
                                std::vector<std::vector<std::vector<int>>> &output){
  pcl::PointCloud<pcl::PointXYZ>::Ptr xyz(new pcl::PointCloud<pcl::PointXYZ>);
  output.clear();
  output.resize(range_image.rows);
  for(auto &it : output){
    it.resize(range_image.cols);
  }
  for (int i = 0; i < range_image.rows; ++ i) {
    for (int j = 0; j < range_image.cols; ++j) {
      pcl::PointXYZ p;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      xyz->push_back(p);
    }
  }
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(xyz);
  std::vector<int> idx(K);
  std::vector<float> dis(K);
  for (int i = 0; i < range_image.rows; ++ i) {
    for (int j = 0; j < range_image.cols; ++j) {
      pcl::PointXYZ p;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      kdtree.nearestKSearch(p, K, idx, dis);
      output.at(i).at(j) = idx;
    }
  }
}
#else
/**
 * @brief 只选一个点周围8个点，以及自己。*/
static void GetNearPoint(const cv::Mat &range_image,
                         const int K,
                         std::vector<std::vector<std::vector<int>>> &output){
  output.clear();
  output.resize(range_image.rows);
  for(auto &it : output){
    it.resize(range_image.cols);
  }
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      auto &T = output.at(i).at(j);

      T.push_back( (i)*range_image.cols + j);
      T.push_back( (i - 1)*range_image.cols + j - 1);
      T.push_back( (i - 1)*range_image.cols + j);
      T.push_back( (i - 1)*range_image.cols + j + 1);

      T.push_back( (i)*range_image.cols + j - 1);
      T.push_back( (i)*range_image.cols + j + 1);

      T.push_back( (i + 1)*range_image.cols + j - 1);
      T.push_back( (i + 1)*range_image.cols + j);
      T.push_back( (i + 1)*range_image.cols + j + 1);
    }
  }

}
#endif


static void FitPlaneSVD(const cv::Mat &range_image, std::vector<int> idx, cv::Vec3f *result){
  Eigen::Matrix<double_t ,Eigen::Dynamic, Eigen::Dynamic> m(3, idx.size());
  double sx(0), sy(0), sz(0);
  for (int i = 0; i < idx.size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx.at(i)];
    sx += m(0, i) = xyz(0);
    sy += m(1, i) = xyz(1);
    sz += m(2, i) = xyz(2);
  }
  sx /= idx.size();
  sy /= idx.size();
  sz /= idx.size();
  Eigen::Matrix<double_t, Eigen::Dynamic, Eigen::Dynamic> q(3, idx.size());
  for (int i = 0; i < idx.size(); ++ i){
    q(0, i) = m(0,i) - sx;
    q(1, i) = m(1,i) - sy;
    q(2, i) = m(2,i) - sz;
  }
  auto X = q*q.transpose();
  Eigen::JacobiSVD<Eigen::Matrix<double_t, Eigen::Dynamic, Eigen::Dynamic> > svd(X, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
  auto U = svd.matrixU();
  auto A = svd.singularValues();
  int min_id=0;
  for (int i=1;i<3;++i) {
    if (A(i) < A(min_id)){
      min_id = i;
    }
  }
  result->operator()(0) = U(0, min_id);
  result->operator()(1) = U(1, min_id);
  result->operator()(2) = U(2, min_id);
  *result /= cv::norm(*result);
  if (result->operator()(2)<0){
    *result = -*result;
  }
}

static void FitPlanePCA(const cv::Mat &range_image, std::vector<int> idx, cv::Vec3f *result){
  Eigen::Matrix<double_t ,Eigen::Dynamic, Eigen::Dynamic> m(3, idx.size());
  double sx(0), sy(0), sz(0);
  for (int i = 0; i < idx.size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx.at(i)];
    sx += m(0, i) = xyz(0);
    sy += m(1, i) = xyz(1);
    sz += m(2, i) = xyz(2);
  }
  sx /= idx.size();
  sy /= idx.size();
  sz /= idx.size();
  Eigen::Matrix<double_t, Eigen::Dynamic, Eigen::Dynamic> q(3, idx.size());
  for (int i = 0; i < idx.size(); ++ i){
    q(0, i) = m(0,i) - sx;
    q(1, i) = m(1,i) - sy;
    q(2, i) = m(2,i) - sz;
  }
  auto X = q*q.transpose();
  Eigen::JacobiSVD<Eigen::Matrix<double_t, Eigen::Dynamic, Eigen::Dynamic> > svd(q, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
  auto U = svd.matrixU();
  auto A = svd.singularValues();
  int min_id=0;
  for (int i=1;i<3;++i) {
    if (A(i) < A(min_id)){
      min_id = i;
    }
  }
  result->operator()(0) = U(0, min_id);
  result->operator()(1) = U(1, min_id);
  result->operator()(2) = U(2, min_id);
  *result /= cv::norm(*result);
  if (result->operator()(2)<0){
    *result = -*result;
  }
}

static void FitVectorSVD(const cv::Mat &range_image, std::vector<int> idx, cv::Vec3f *result){
  Eigen::Matrix<double_t ,Eigen::Dynamic, Eigen::Dynamic> q(3, idx.size() - 1);
  for (int i = 1; i < idx.size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx.at(i)];
    q(0, i - 1) = xyz(0) - ((cv::Vec3f*)range_image.data)[idx.at(0)](0);
    q(1, i - 1) = xyz(1) - ((cv::Vec3f*)range_image.data)[idx.at(0)](1);
    q(2, i - 1) = xyz(2) - ((cv::Vec3f*)range_image.data)[idx.at(0)](2);
  }
  Eigen::JacobiSVD<Eigen::Matrix<double_t, Eigen::Dynamic, Eigen::Dynamic> > svd(q, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
  auto U = svd.matrixU();
  auto A = svd.singularValues();
  int min_id=0;
  for (int i=1;i<3;++i) {
    if (A(i) < A(min_id)){
      min_id = i;
    }
  }
  result->operator()(0) = U(0, min_id);
  result->operator()(1) = U(1, min_id);
  result->operator()(2) = U(2, min_id);
  *result /= cv::norm(*result);
  if (result->operator()(2)<0){
    *result = -*result;
  }
}

static void PlaneSVD(const cv::Mat &range_image, int K, cv::Mat &result){
  result.create(range_image.rows,range_image.cols, CV_32FC3);//3 通道，非常make sense
  std::vector<std::vector<std::vector<int>>> cor;
  cor.resize(range_image.rows);
  GetNearPoint(range_image, K, cor);
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      pcl::PointXYZ p;
      cv::Vec3f res;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      FitPlaneSVD(range_image, cor.at(i).at(j), &res);
      result.at<cv::Vec3f>(i, j) = res;
    }
  }
}

static void PlanePCA(const cv::Mat &range_image, int K, cv::Mat &result){
  result.create(range_image.rows,range_image.cols, CV_32FC3);//3 通道，非常make sense
  std::vector<std::vector<std::vector<int>>> cor;
  GetNearPoint(range_image, K, cor);
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      pcl::PointXYZ p;
      cv::Vec3f res;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      FitPlanePCA(range_image, cor.at(i).at(j), &res);
      result.at<cv::Vec3f>(i, j) = res;
    }
  }
}

static void VectorSVD(const cv::Mat &range_image, int K, cv::Mat &result){
  result.create(range_image.rows,range_image.cols, CV_32FC3);//3 通道，非常make sense
  std::vector<std::vector<std::vector<int>>> cor;
  GetNearPoint(range_image, K, cor);
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      pcl::PointXYZ p;
      cv::Vec3f res;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      FitVectorSVD(range_image, cor.at(i).at(j), &res);
      result.at<cv::Vec3f>(i, j) = res;
    }
  }
}

static void FitQuadSVD(const cv::Mat &range_image, std::vector<int> idx, cv::Vec3f *result){
  Eigen::Matrix<double_t ,Eigen::Dynamic, Eigen::Dynamic> m(10, idx.size());
  double s0(0), s1(0), s2(0), s3(0), s4(0), s5(0), s6(0), s7(0), s8(0), s9(0);
  for (int i = 0; i < idx.size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx.at(i)];
    s0 += m(0, i) = xyz(0) * xyz(0);
    s1 += m(1, i) = xyz(1) * xyz(1);
    s2 += m(2, i) = xyz(2) * xyz(2);
    s3 += m(3, i) = xyz(0) * xyz(1);
    s4 += m(4, i) = xyz(0) * xyz(2);
    s5 += m(5, i) = xyz(1) * xyz(2);
    s6 += m(6, i) = xyz(0);
    s7 += m(7, i) = xyz(1);
    s8 += m(8, i) = xyz(2);
    s9 += 1;
  }
  Eigen::JacobiSVD<Eigen::Matrix<double_t, Eigen::Dynamic, Eigen::Dynamic> > svd(m.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV); // ComputeThinU | ComputeThinV
  auto U = svd.matrixU();
  auto V = svd.matrixV();
  auto A = svd.singularValues();
  double x, y, z;
  x = ((cv::Vec3f*)range_image.data)[idx.at(0)](0);
  y = ((cv::Vec3f*)range_image.data)[idx.at(0)](1);
  z = ((cv::Vec3f*)range_image.data)[idx.at(0)](2);
  result->operator()(0) = 2*V(0, 8) * x + V(3, 8) * y + V(4, 8) * z + V(6, 8);
  result->operator()(1) = 2*V(1, 8) * y + V(3, 8) * x + V(5, 8) * z + V(7, 8);
  result->operator()(2) = 2*V(2, 8) * z + V(4, 8) * x + V(5, 8) * y + V(8, 8);
  *result /= cv::norm(*result);
  if (result->operator()(2)<0){
    *result = -*result;
  }
}

static void FitQuadTransSVD(const cv::Mat &range_image, std::vector<int> idx, cv::Vec3f *result){
  Eigen::Matrix<double_t ,Eigen::Dynamic, Eigen::Dynamic> m(3, idx.size());
  std::vector<double> sx, sy, sz;
  cv::Vec3f p = ((cv::Vec3f*)range_image.data)[idx.at(0)];
  for (int i = 0; i < idx.size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx.at(i)];
    sx.push_back(xyz(0));
    sy.push_back(xyz(1));
    sz.push_back(xyz(2));
  }
  //先搞个JB VectorSVD弄个，哎，坑爹
  //按照某种方法排序?
  cv::Vec3f core;
  FitVectorSVD(range_image, idx, &core);
  core /= cv::norm(core);
  cv::Vec3f n=(core + cv::Vec3f(0,0,1))/2;
  Eigen::Vector3d N(n(0), n(1), n(2));
  N = N / N.norm();
  Eigen::Vector3d T;
  Eigen::Matrix3d R = 2 * N * N.transpose() - Eigen::Matrix3d::Identity();
  for (int i = 0; i < idx.size(); ++ i){
    m(0, i) = sx.at(i);
    m(1, i) = sy.at(i);
    m(2, i) = sz.at(i);
  }
  T(0) = - m(0, 0);
  T(1) = - m(1, 0);
  T(2) = - m(2, 0);
  for (int i = 0; i < idx.size(); ++ i){
    Eigen::Vector3d v;
    v(0) = m(0, i) + T(0);
    v(1) = m(1, i) + T(1);
    v(2) = m(2, i) + T(2);
    v = R * v;
    m(0, i) = v(0);
    m(1, i) = v(1);
    m(2, i) = v(2);
    sx.at(i) = v(0);
    sy.at(i) = v(1);
    sz.at(i) = v(2);
  }
  Eigen::Matrix<double_t ,Eigen::Dynamic, 1> b(idx.size());
  Eigen::Matrix<double_t ,Eigen::Dynamic, Eigen::Dynamic> q(6, idx.size());
  for (int i = 0; i < idx.size(); ++ i){
    b(i) = sz.at(i);
    q(0, i) = sx[i] * sx[i];
    q(1, i) = sy[i] * sy[i];
    q(2, i) = sx[i] * sy[i];
    q(3, i) = sx[i];
    q(4, i) = sy[i];
    q(5, i) = 1;
  }
  Eigen::Matrix<double, 6, 1> x = q.transpose().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  Eigen::Vector3d v(x(3), x(4), 1);
  v = (R.inverse()*v);
  v = v / v.norm();
  result->operator()(0) = v(0);
  result->operator()(1) = v(1);
  result->operator()(2) = v(2);
}

static void QuadSVD(const cv::Mat &range_image, int K, cv::Mat &result){
  result.create(range_image.rows,range_image.cols, CV_32FC3);//3 通道，非常make sense
  std::vector<std::vector<std::vector<int>>> cor;
  GetNearPoint(range_image, K, cor);
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      pcl::PointXYZ p;
      cv::Vec3f res;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      FitQuadSVD(range_image, cor.at(i).at(j), &res);
      result.at<cv::Vec3f>(i, j) = res;
    }
  }
}

static void QuadTransSVD(const cv::Mat &range_image, int K, cv::Mat &result){
  result.create(range_image.rows,range_image.cols, CV_32FC3);//3 通道，非常make sense
  std::vector<std::vector<std::vector<int>>> cor;
  GetNearPoint(range_image, K, cor);
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      pcl::PointXYZ p;
      cv::Vec3f res;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      FitQuadTransSVD(range_image, cor.at(i).at(j), &res);
      result.at<cv::Vec3f>(i, j) = res;
    }
  }
}

static void SortIdx(const cv::Mat &range_image, std::vector<int> *idx){
  Eigen::Matrix<double_t ,Eigen::Dynamic, Eigen::Dynamic> m(3, idx->size());
  std::vector<double> sx, sy, sz;
  cv::Vec3f p = ((cv::Vec3f*)range_image.data)[idx->at(0)];
  for (int i = 0; i < idx->size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx->at(i)];
    sx.push_back(xyz(0));
    sy.push_back(xyz(1));
    sz.push_back(xyz(2));
  }
  cv::Vec3f core;
  FitVectorSVD(range_image, *idx, &core);
  core /= cv::norm(core);
  cv::Vec3f n=(core + cv::Vec3f(0,0,1))/2;
  Eigen::Vector3d N(n(0), n(1), n(2));
  N = N / N.norm();
  Eigen::Vector3d T;
  Eigen::Matrix3d R = 2 * N * N.transpose() - Eigen::Matrix3d::Identity();
  for (int i = 0; i < idx->size(); ++ i){
    m(0, i) = sx.at(i);
    m(1, i) = sy.at(i);
    m(2, i) = sz.at(i);
  }
  T(0) = - m(0, 0);
  T(1) = - m(1, 0);
  T(2) = - m(2, 0);
  for (int i = 0; i < idx->size(); ++ i){
    Eigen::Vector3d v;
    v(0) = m(0, i) + T(0);
    v(1) = m(1, i) + T(1);
    v(2) = m(2, i) + T(2);
    v = R * v;
    m(0, i) = v(0);
    m(1, i) = v(1);
    m(2, i) = v(2);
    sx.at(i) = v(0);
    sy.at(i) = v(1);
    sz.at(i) = v(2);
  }
  for (int i = 1; i< idx->size(); ++ i) {
    for (int j = i + 1; j < idx->size(); ++ j){
      if (atan2(sy[i], sx[i]) > atan2(sy[j], sx[j])){
        std::swap(sx[i], sx[j]);
        std::swap(sy[i], sy[j]);
        std::swap(idx->operator[](i), idx->operator[](j));
      }

    }
  }
}

//按照面积权重
static void FitAreaWeighted(const cv::Mat &range_image, std::vector<int> idx, cv::Vec3f *result){
  std::vector<cv::Vec3f> s;
  cv::Vec3f p = ((cv::Vec3f*)range_image.data)[idx.at(0)];
  cv::Vec3f tot(0,0,0);
  idx.at(0) = idx.at(idx.size()-1);
  for (int i = 0; i < idx.size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx.at(i)];
    s.push_back(xyz);
  }
  for (int i = 1; i < idx.size(); ++ i) {
    cv::Vec3f v = (s[i] - p).cross(s[i-1] - p);
    double w = cv::norm(v)/2;
    v = v / cv::norm(v);
    if (isnan(v(0))){
      continue;
    }
    tot += v * w;
  }
  tot = tot / cv::norm(tot);
  *result = tot;
}

//按照面积权重
static void FitAngleWeighted(const cv::Mat &range_image, std::vector<int> idx, cv::Vec3f *result){
  std::vector<cv::Vec3f> s;
  cv::Vec3f p = ((cv::Vec3f*)range_image.data)[idx.at(0)];
  cv::Vec3f tot(0,0,0);
  idx.at(0) = idx.at(idx.size()-1);
  for (int i = 0; i < idx.size(); ++ i){
    cv::Vec3f xyz=((cv::Vec3f*)range_image.data)[idx.at(i)];
    s.push_back(xyz);
  }
  for (int i = 1; i < idx.size(); ++ i) {
    cv::Vec3f v = (s[i] - p).cross(s[i-1] - p);
    double tmp =   ((s[i] - p).dot(s[i-1] - p)) / (cv::norm(s[i] - p) * cv::norm(s[i-1]-p));
    if (tmp>1) tmp=1;
    if (tmp<-1) tmp=-1;
    double w = acos(tmp);
    v = v / cv::norm(v);
    if (isnan(v(0))){
      continue;
    }
    tot += v * w;
  }
  tot = tot / cv::norm(tot);
  *result = tot;
}

static void WeightingFactorArea(const cv::Mat &range_image, int K, cv::Mat &result){
  result.create(range_image.rows,range_image.cols, CV_32FC3);//3 通道，非常make sense
  std::vector<std::vector<std::vector<int>>> cor;
  GetNearPoint(range_image, K, cor);
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      pcl::PointXYZ p;
      cv::Vec3f res;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      SortIdx(range_image, &cor.at(i).at(j));
      FitAreaWeighted(range_image, cor.at(i).at(j), &res);
      result.at<cv::Vec3f>(i, j) = res;
    }
  }
}

static void WeightingFactorAngle(const cv::Mat &range_image, int K, cv::Mat &result){
  result.create(range_image.rows,range_image.cols, CV_32FC3);//3 通道，非常make sense
  std::vector<std::vector<std::vector<int>>> cor;
  GetNearPoint(range_image, K, cor);
  for (int i = 1; i < range_image.rows - 1; ++ i) {
    for (int j = 1; j < range_image.cols - 1; ++j) {
      pcl::PointXYZ p;
      cv::Vec3f res;
      p.x = range_image.at<cv::Vec3f>(i, j)(0);
      p.y = range_image.at<cv::Vec3f>(i, j)(1);
      p.z = range_image.at<cv::Vec3f>(i, j)(2);
      SortIdx(range_image, &cor.at(i).at(j));
      FitAngleWeighted(range_image, cor.at(i).at(j), &res);
      result.at<cv::Vec3f>(i, j) = res;
    }
  }
}

void GetNormal(const cv::Mat &range_image, const int &K, cv::Mat *result, const NEAREST_METHOD& METHOD){
  if (K < 3) {
    std::cerr <<"haha" << std::endl;
    exit(-1);
  }
  if (METHOD == PLANESVD){
    PlaneSVD(range_image, K, *result);
  }
  if (METHOD == PLANEPCA){
    PlanePCA(range_image, K, *result);
  }
  if (METHOD == VECTORSVD){

    VectorSVD(range_image, K, *result);
  }
  if (METHOD == QUADSVD){
    if (K < 10) {
      std::cerr<<"the quad svd algorithm needs K>=10, more details you can see in the paper."<<std::endl;
      exit(-1);
    }
    QuadSVD(range_image, K, *result);
  }
  if (METHOD == QUADTRANSSVD){
    if (K < 10) {
      std::cerr<<"the quad  trans svd algorithm needs K>=6, more details you can see in the paper."<<std::endl;
      exit(-1);
    }
    QuadTransSVD(range_image, K, *result);
  }
  if (METHOD == AREAWEIGHTED){
   WeightingFactorArea(range_image, K, *result);
  }
  if (METHOD == ANGLEWEIGHTED){
    WeightingFactorAngle(range_image, K, *result);
  }
}


