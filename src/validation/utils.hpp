#ifndef VALIDATION_UTILS_
#define VALIDATION_UTILS_
#include<vector>
#include<cmath>
#include<iostream>
#include<algorithm>
#include"../global.h"
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
//get TP and TN
void inline GetTpAndTn(const std::vector<cxxnet::real_t>& distances, const std::vector<bool>& pair_labels, cxxnet::real_t threshold,int* tp,int* tn,int * fp,int* fn){
  for (size_t i = 0; i<distances.size(); i++){
    using namespace std;
    //cout <<"abc "<< distances[i] << " " << pair_labels[i] << endl;
    *tp = 0, *tn = 0, *fp = 0, *fn = 0;
    if (distances[i]>threshold){
      if (pair_labels[i])(*tp)++;
      else (*fp)++;
    }
    else{
      if (!pair_labels[i])(*tn)++;
      else (*fn)++;
    }
  }
}
void GetThreshold(const std::vector<cxxnet::real_t>& distances, const std::vector<bool>& labels,cxxnet::real_t max_far,cxxnet::real_t* result){
  using namespace std;
  int count_p = 0, count_n = 0;
  for (size_t i = 0; i < labels.size(); i++){
    if (labels[i]){
      count_p++;
    }
  }
  count_n = labels.size() - count_p;
  int tp, fp, tn, fn;
  cxxnet::real_t threshold;
  cxxnet::real_t max_accuracy = 0;
  cxxnet::real_t max_tpr = 0;
  vector<cxxnet::real_t> temp_distances(distances);
  sort(temp_distances.begin(), temp_distances.end());
  for (size_t i = temp_distances.size()-1; i>=0; i--){
    threshold = temp_distances[i];
    GetTpAndTn(distances,labels, threshold,&tp,&tn,&fp,&fn);
    if (max_far == -1){
      cxxnet::real_t accuracy = (tp + tn)*1.0 / labels.size();
      if (accuracy > max_accuracy){
        *result = threshold;
        max_accuracy = accuracy;
      }
    }
    else{
      cxxnet::real_t far, tpr;
      far = fp*1.0 / (count_n);
      tpr = tp*1.0 / (count_p);
      if (far <= max_far&&tpr > max_tpr){
        max_tpr = tpr;
        *result = threshold;
      }
      if (far > max_far)break;
    }
  }
}
#endif