#ifndef CXXNET_CLOSEDSET_RECOGNITION_VALIDATION_
#define CXXNET_CLOSEDSET_RECOGNITION_VALIDATION_
#include<vector>
#include<cmath>
#include<iostream>
#include"utils.hpp"
#include"distance.hpp"
cxxnet::real_t ClosedsetRecognitionValidation(const std::vector<std::vector<cxxnet::real_t> >& features, const std::vector<std::vector<cxxnet::real_t> >&labels,cxxnet::real_t max_far){
  using namespace std;
  vector<vector<cxxnet::real_t> > gallery_features, probe_features;
  vector<cxxnet::real_t> gallery_labels,probe_labels;
  for (size_t i = 0; i < features.size(); i++){
    if (labels[i][0] >= 1000){//gallery
      gallery_features.push_back(features[i]);
      gallery_labels.push_back(labels[i][0] - 1000);
    }
    else{
      probe_features.push_back(features[i]);
      probe_labels.push_back(labels[i][0]);
    }
  }
  vector<cxxnet::real_t> distances;
  vector<bool> pair_labels;
  for (size_t i = 0; i < gallery_features.size(); i++){
    for (size_t j = 0; j < probe_features.size(); j++){
      distances.push_back(CosDistance(gallery_features[i], probe_features[j]));
      pair_labels.push_back((gallery_labels[i] == probe_labels[j]));
    }
  }
  //cout << "probe" << probe_features.size() << " gallery:" << gallery_features.size() << endl;
  cxxnet::real_t threshold,tpr;
  GetThreshold(distances, pair_labels, max_far, &threshold);
  int tp, tn, fp, fn;
  GetTpAndTn(distances, pair_labels, threshold, &tp, &tn, &fp, &fn);
  tpr = tp*1.0 / pair_labels.size();
  return tpr;
}
#endif
