#include <bitset>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "mgrit_solver.h"
#include "neural_network.h"
#include "typedefs.h"
 
using namespace Eigen;
using namespace std;

//generates the data used for the four layer problem
vector<MatrixXd> generateData()
{

  srand(2);                  // seed the random number generator to get the same data each time
  const int bit = 12;
  const int num_of_training_data = 500;
  unsigned int max_num = pow(2, bit);  // maximum number that can be stored given the bit size
  
  Matrix<double, num_of_training_data, bit * 2> input;
  Matrix<double, num_of_training_data, bit> target;

  for (int i =  0; i < num_of_training_data; i++)
  {
    // generate two random numbers and calculate the binary sum
    unsigned int number_1 = rand() % max_num;
    unsigned int number_2 = rand() % max_num;
    unsigned int total = (number_1 + number_2) % max_num;

    // convert the numbers and their sum to binary and store the result
    for (int j = 0; j < bit; j++)
    {
      input(i,bit-j-1) = (number_1 >> j) & 1;
    }
    for (int j = 0; j < bit; j++)
    {
      input(i,2*bit-j-1) = (number_2 >> j) & 1;
    }
    for (int j = 0; j < bit; j++)
    {
      target(i,bit-j-1) = (total >> j) & 1;
    }
  }

  return {input, target};
}

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
  
  const int N = 6400;                         // Number of training steps
  const int m = 2;                          // Coarsening factor
  const int max_level = 10;                  // The maximum level the MGRIT algorithm recurses to
  const float alpha_b = 0.025;                // The learning rate of the neural network on the fine grid
  const float alpha_max = 0.2;              // The maximum learning rate the algorithm can increase to on the coarse grids
  const bool serialized_training = true;   // If true, the nn trains in a serial fashion, otherwise it trains in a batch fashion
  const bool f_cycles = true;              // Determine whether to run F cycles (if false, the algorithm runs V cycles)
  const bool display_output = true;         // Displays stats about the MGRIT algorithm as it is running

  // construct the nn
  vector<MatrixXd> generatedData = generateData();
  MatrixXd input = generatedData[0];
  MatrixXd target = generatedData[1];
  NeuralNetwork four_layer_nn(alpha_b, {MatrixXd::Random(24,128), MatrixXd::Random(128,64), MatrixXd::Random(64,12)});

  // initialize the weights and the right hand side of the linear equation that MGRIT solves
  listOfWeights initial_weights(N+1, {MatrixXd::Zero(24,128), MatrixXd::Zero(128,64), MatrixXd::Zero(64,12)});
  initial_weights[0] = four_layer_nn.getWeights();
  listOfWeights rhs = initial_weights;

  // construct the phi functions used in the MGRIT algorithm
  vector<vector<phiFuncType>> phis = generatePhis(four_layer_nn, input, target, alpha_b, alpha_max, max_level, serialized_training);

  // construct the MGRIT solver and run it
  MGRITSolver solver(m, phis, max_level, display_output);
  listOfWeights MGRIT_weights = solver.run(initial_weights, rhs, pow(10, -9) * sqrt(N+1), f_cycles);

  weightType MGRIT_last_weights = MGRIT_weights[MGRIT_weights.size()-1];
  MatrixXd MGRIT_last_weight = MGRIT_last_weights[MGRIT_last_weights.size()-1].row(53);
  cout << MGRIT_last_weight << endl;

  four_layer_nn.setWeights(rhs[0]);

  for (int i = 0; i < N; i++)
  {
    serialized_training? four_layer_nn.train(input.row(i % input.rows()), target.row(i % target.rows())) : four_layer_nn.train(input, target);
  }

  MatrixXd last_weight = four_layer_nn.getWeights()[four_layer_nn.getWeights().size()-1].row(53);
  cout << last_weight << endl;

}