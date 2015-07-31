#include<string>
#include<sstream>
#include"pairwise_cross_validation.hpp"
#include"closedset_recognition_validation.hpp"
#include"global.h"
const int validation_types_count = 2;
const std::string validation_types[] = {
  "pairwise_cross",
  "closedset_recognition"
};
size_t fold;
std::string ValidationFunction(const std::vector<std::vector<cxxnet::real_t> >& features, const std::vector<std::vector<cxxnet::real_t> >&labels, const std::string& validation_type){
  using namespace std;
  stringstream ss;
  if (validation_type == "pairwise_cross"){
    size_t fold = 10;
    cxxnet::real_t precision = PairwiseCrossValidation(features, labels, fold);
    std::stringstream ss;
    ss << "-pairwize-" << fold << ":" << precision;
  }
  else if (validation_type == "closedset_recognition"){
    cxxnet::real_t max_far,tpr_100,tpr_1000,tpr_10000;
    max_far = 0.01;
    tpr_100 = ClosedsetRecognitionValidation(features, labels, max_far);
    max_far = 0.001;
    tpr_1000 = ClosedsetRecognitionValidation(features, labels, max_far);
    max_far = 0.0001;
    tpr_1000 = ClosedsetRecognitionValidation(features, labels, max_far);
    std::stringstream ss;
    ss << "-closedset:" << tpr_100<<"|"<<tpr_1000<<"|"<<tpr_10000;
  }
  return ss.str();
}