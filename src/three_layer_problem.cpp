#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "mgrit_solver.h"
#include "neural_network.h"
#include "typedefs.h"
 
using namespace Eigen;
using namespace std;

// generates the phis to use on each level of the grid
vector<vector<phiFuncType>> generatePhis(NeuralNetwork& network, const MatrixXd& my_input, const MatrixXd& my_target, const float& base_alpha, const float& max_alpha, const unsigned int& max_level, const bool& serialized)
{

  vector<vector<phiFuncType>> phis;
  int max_iter = serialized? my_input.rows() : 1;
  for (int i = 0; i < max_level; i++)
  {
    vector<phiFuncType> row_phis;
    for (int j = 0; j < max_iter; j++)
    {
      // set the training input and target for the nn
      MatrixXd input = serialized? my_input.row(j) : my_input;
      MatrixXd target = serialized? my_target.row(j) : my_target;;

      // set the alpha level
      float alpha = base_alpha * pow(2,i);
      if (alpha > max_alpha) {alpha = max_alpha;};

      row_phis.push_back( [&network, input, target, alpha](const weightType &weights)
                          {
                            network.setAlpha(alpha);
                            network.setWeights(weights);
                            network.train(input, target);
                            return network.getWeights();
                          }
                        );
    }
  phis.push_back(row_phis);
  }

  return phis;
}

int main(int argc, char* argv[])
{

  const int N = 100;                        // Number of training steps
  const int m = 2;                          // Coarsening factor
  const int max_level = 10;                 // The maximum level the MGRIT algorithm recurses to
  const float alpha_b = 0.1;                // The learning rate of the neural network on the fine grid
  const float alpha_max = 30.0;             // The maximum learning rate the algorithm can increase to on the coarse grids
  const bool serialized_training = true;    // If true, the nn trains in a serial fashion, otherwise it trains in a batch fashion
  const bool f_cycles = true;               // Determine whether to run F cycles (if false, the algorithm runs V cycles)
  const bool display_output = true;         // Displays stats about the MGRIT algorithm as it is running

  // construct the nn
  Matrix<double, 4, 3> input;
  input << 0.0, 0.0, 1.0,
           0.0, 1.0, 1.0,
           1.0, 0.0, 1.0,
           1.0, 1.0, 1.0;

  Matrix<double, 4, 1> target;
  target << 0.0,
  			    1.0,
  			    1.0,
  			    0.0;
  NeuralNetwork three_layer_nn(alpha_b, {MatrixXd::Random(3,4), MatrixXd::Random(4,1)});

  // initialize the weights and the right hand side of the linear equation that MGRIT solves
  listOfWeights initial_weights(N+1, {MatrixXd::Zero(3,4), MatrixXd::Zero(4,1)});
  initial_weights[0] = three_layer_nn.getWeights();
  listOfWeights rhs = initial_weights;

  // construct the phi functions used in the MGRIT algorithm
  vector<vector<phiFuncType>> phis = generatePhis(three_layer_nn, input, target, alpha_b, alpha_max, max_level, serialized_training);
  
  // construct the MGRIT solver and run it
  MGRITSolver solver(m, phis, max_level, display_output);
  listOfWeights MGRIT_weights = solver.run(initial_weights, rhs, pow(10, -9) * sqrt(N+1), f_cycles);

  cout << "The MGRIT trained weights are : " << endl;
  for (MatrixXd weight : three_layer_nn.getWeights())
  {
  	cout << weight << endl;
  }

  // train the neural network in a sequential fashion to get the expected set of final weights
  three_layer_nn.setWeights(initial_weights[0]);
  for (int i = 0; i < N; i++)
  {
    serialized_training? three_layer_nn.train(input.row(i % input.rows()), target.row(i % target.rows())) : three_layer_nn.train(input, target);
  }

  cout << "The actual values of the trained weights are: " << endl;
  for (MatrixXd weight : MGRIT_weights[MGRIT_weights.size()-1])
  {
    cout << weight << endl;
  }

}