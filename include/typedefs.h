#ifndef HH_TYPEDEFS_HH
#define HH_TYPEDEFS_HH

#include <Eigen/Dense>
#include <functional>
#include <vector>

using namespace Eigen;
using namespace std;

typedef vector<MatrixXd> weightType;
typedef vector<weightType> listOfWeights;
typedef function<weightType (const weightType& weights)> phiFuncType;

#endif