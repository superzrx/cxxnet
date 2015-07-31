#ifndef VALIDATION_DISTANCE_
#define VALIDATION_DISTANCE_
#include<vector>
#include<cmath>
#include<iostream>
#include"global.h"


inline cxxnet::real_t DotProduct(const std::vector<cxxnet::real_t>& feature1, const std::vector<cxxnet::real_t>& feature2){
	cxxnet::real_t ret = 0;
	for (size_t i = 0; i < feature1.size(); i++){
		ret += feature1[i] * feature2[i];
	}
	return ret;
}
inline cxxnet::real_t L2Distance(const std::vector<cxxnet::real_t>& feature1,const std::vector<cxxnet::real_t>& feature2){
	using namespace std;
	cxxnet::real_t dis = 0;
	for (size_t i = 0; i < feature1.size(); i++){
		dis += pow(feature1[i] - feature2[i],2);
	}
	return sqrt(dis);
}
inline cxxnet::real_t CosDistance(const std::vector<cxxnet::real_t>& feature1, const std::vector<cxxnet::real_t>& feature2){
	using namespace std;
	cxxnet::real_t d, d1, d2;//distance
	d = DotProduct(feature1, feature2);
	d1 = DotProduct(feature1, feature1);
	d2 = DotProduct(feature2, feature2);
	return d / sqrt(d1*d2);
}
#endif