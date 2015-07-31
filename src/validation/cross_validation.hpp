#ifndef CXXNET_CROSS_VALIDATION_
#define CXXNET_CROSS_VALIDATION_
#include<vector>
#include<cmath>
#include<iostream>
#include"distance.hpp"
#include"global.h"
#include<algorithm>
cxxnet::real_t MaxValue(const std::vector<cxxnet::real_t>& feature){
	cxxnet::real_t maxVal = feature[0];
	for (size_t i = 0; i < feature.size(); i++){
		maxVal = feature[i]>maxVal ? feature[i] : maxVal;
	}
	return maxVal;
}
cxxnet::real_t MinValue(const std::vector<cxxnet::real_t>& feature){
	cxxnet::real_t minVal = feature[0];
	for (size_t i = 0; i < feature.size(); i++){
		minVal = feature[i]<minVal ? feature[i] : minVal;
	}
	return minVal;
}
cxxnet::real_t GetPresicion(const std::vector<cxxnet::real_t>& distances, const std::vector<bool>& pair_labels, cxxnet::real_t threshold){
	int correct_count = 0;
	for (size_t i = 0; i<distances.size(); i++){
    using namespace std;
    //cout <<"abc "<< distances[i] << " " << pair_labels[i] << endl;
		if (distances[i]>threshold){
			if (pair_labels[i])correct_count++;
		}
		else{
			if (!pair_labels[i])correct_count++;
		}
	}
	return correct_count*1.0F/ distances.size();
}
cxxnet::real_t GetThreshold(const std::vector<cxxnet::real_t>& distances, const std::vector<bool>& pair_labels){
  using namespace std;
	int pair_same_num = 0;
	for (size_t i = 0; i < pair_labels.size(); i++){
		if (pair_labels[i]){
			pair_same_num++;
		}
	}
	cxxnet::real_t threshold, precision;
    //min_distance,max_distance;
    //max_distance = MaxValue(distances);
	//min_distance = MinValue(distances);
	cxxnet::real_t max_precision = 0;
	cxxnet::real_t max_threshold = 0;
  vector<cxxnet::real_t> temp_distances(distances);
  sort(temp_distances.begin(),temp_distances.end());
  if (temp_distances.size() <= 100){
    for (size_t i = 0; i < temp_distances.size(); i++){
      threshold = temp_distances[i];
      precision = GetPresicion(distances, pair_labels, threshold);
      if (precision > max_precision){
        max_precision = precision;
        max_threshold = threshold;
      }
    }
  }
  else{
    for (size_t i = 0; i < 100; i++){
      threshold = temp_distances[i*temp_distances.size()/100];
      precision = GetPresicion(distances, pair_labels, threshold);
      if (precision > max_precision){
        max_precision = precision;
        max_threshold = threshold;
      }
    }
  }
	return max_threshold;
}

cxxnet::real_t CrossValidation(const std::vector<std::vector<cxxnet::real_t> >& features, const std::vector<std::vector<cxxnet::real_t> >&label, size_t fold){
	using namespace std;
	vector<cxxnet::real_t> distances;
	vector<bool> pair_labels;
	for (size_t i = 0; i < features.size(); i += 2){
		distances.push_back(CosDistance(features[i], features[i + 1]));
		pair_labels.push_back((label[i] == label[i + 1]));
	}
	int bin = distances.size() / fold;
	cxxnet::real_t sum_precision=0;
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
		cxxnet::real_t threshold = GetThreshold(train_distances, train_pair_labels);
		cxxnet::real_t precision = GetPresicion(test_distances, test_pair_labels, threshold);
    //cout << threshold << " " << precision << endl;
		sum_precision +=precision;
	}
	return sum_precision/fold;
}

#endif
