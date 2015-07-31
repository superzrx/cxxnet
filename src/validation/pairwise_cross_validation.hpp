#ifndef CXXNET_PAIRWISE_CROSS_VALIDATION_
#define CXXNET_PAIRWISE_CROSS_VALIDATION_
#include<vector>
#include<cmath>
#include<iostream>
#include"utils.hpp"
#include"distance.hpp"
#include"global.h"
cxxnet::real_t PairwiseCrossValidation(const std::vector<std::vector<cxxnet::real_t> >& features, const std::vector<std::vector<cxxnet::real_t> >&label, size_t fold){
	using namespace std;
	vector<cxxnet::real_t> distances;
	vector<bool> pair_labels;
	for (size_t i = 0; i < features.size(); i += 2){
		distances.push_back(CosDistance(features[i], features[i + 1]));
		pair_labels.push_back((label[i] == label[i + 1]));
	}
	int bin = distances.size() / fold;
	cxxnet::real_t sum_accuracy=0;
	for (size_t i = 0; i < fold; i++){
		vector<cxxnet::real_t> train_distances, test_distances;
		vector<bool> train_pair_labels, test_pair_labels;
		size_t start = i*bin;
		size_t end = start + bin;
		end = end >distances.size()?distances.size():end;
		for (size_t j = 0; j < distances.size(); j++){
			if (j >= start&&j < end){
				test_distances.push_back(distances[j]);
				test_pair_labels.push_back(pair_labels[j]);
			}
			else{
				train_distances.push_back(distances[j]);
				train_pair_labels.push_back(pair_labels[j]);
			}
		}
    cxxnet::real_t threshold, accuracy; 
    int tp, tn, fp, fn;
    GetThreshold(train_distances, train_pair_labels,-1,&accuracy);
		GetTpAndTn(test_distances, test_pair_labels, threshold,&tp,&tn,&fp,&fn);
    accuracy = (tp + tn)*1.0 / test_pair_labels.size();
    sum_accuracy += accuracy;
	}
  return sum_accuracy / fold;
}

#endif
