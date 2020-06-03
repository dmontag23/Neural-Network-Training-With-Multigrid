#ifndef HH_NEURAL_NETWORK_HH
#define HH_NEURAL_NETWORK_HH

#include <Eigen/Dense>
#include <stack>
#include <vector>

#include "typedefs.h"

using namespace Eigen;
using namespace std;

// Neural Network Class
class NeuralNetwork {

   private:

   	// private class data
      float alpha;              // the learning rate of the nn
      weightType weights;

      // private methods
      MatrixXd sigmoid(const MatrixXd& x, const bool& derivative) const;
   
   public:

   	// class constructor
      NeuralNetwork(float my_alpha, weightType my_weights);

      // getters and setters
      float getAlpha() const;
      weightType getWeights() const;
      void setAlpha(float my_alpha);
      void setWeights(weightType my_weights);

      // public methods
      void train(const MatrixXd& input, const MatrixXd& target);
};

#endif