//
// Created by bohuan on 2019/11/4.
//

#ifndef PCNML_INCLUDE_PCNML_THIRDALGORITHM_H_
#define PCNML_INCLUDE_PCNML_THIRDALGORITHM_H_


#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>

/* All of the methods you can see in
 * 'Comparison of Surface Normal Estimation
 *  Methods for Range Sensing Applications'.
 * **/
enum NEAREST_METHOD{
PLANESVD,
PLANEPCA,
VECTORSVD,
QUADSVD,
QUADTRANSSVD,
AREAWEIGHTED,
ANGLEWEIGHTED
};

/**
 * @brief calculate the normal use different method
 * @param[in] range_image the input range image
 * @param[out] result the output result
 * @param[in] METHOD which method will be used in the function
 * @param[in] K because all of the method is base on KNN, so need the K. note: the K include the the point itself.
 * for example: if the K = 2, for point p, it will get 2 closet point from the range_image, which is p and p'.
 * you can see the p is in the set.*/
void GetNormal(const cv::Mat &range_image, const int &K, cv::Mat *result, const NEAREST_METHOD& METHOD);

#endif //PCNML_INCLUDE_PCNML_THIRDALGORITHM_H_
