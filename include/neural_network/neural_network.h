#ifndef HH_NEURAL_NETWORK_HH
#define HH_NEURAL_NETWORK_HH

#include <Eigen/Dense>
#include <stack>
#include <vector>

#include "typedefs.h"

using namespace Eigen;
using namespace std;

class NeuralNetwork {

   private:

      float alpha;              // the learning rate of the nn
      weightType weights;       // weights of the network

      MatrixXd sigmoid(const MatrixXd& x, const bool& derivative) const;
   
   public:

      NeuralNetwork(float my_alpha, weightType my_weights);

      // getters and setters
      float getAlpha() const;
      weightType getWeights() const;
      void setAlpha(float my_alpha);
      void setWeights(weightType my_weights);

      void train(const MatrixXd& input, const MatrixXd& target);
      
};

#endif