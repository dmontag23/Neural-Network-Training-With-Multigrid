#include "neural_network.h"

float NeuralNetwork::getAlpha() const
{
   return alpha;
}

weightType NeuralNetwork::getWeights() const
{
   return weights;
}

NeuralNetwork::NeuralNetwork(float my_alpha, weightType my_weights) : alpha{my_alpha}, weights{my_weights}{}

void NeuralNetwork::setAlpha(float my_alpha)
{
   alpha = my_alpha;
}

void NeuralNetwork::setWeights(weightType my_weights)
{
   weights = my_weights;
}

MatrixXd NeuralNetwork::sigmoid(const MatrixXd& x, const bool& derivative) const
{
   if (derivative)
   {
      return x.array() * (1.0 - x.array());
   }
   else
   {
      return 1.0 / (1.0 + (-1.0 * x).array().exp());
   }
}

void NeuralNetwork::train(const MatrixXd& input, const MatrixXd& target)
{
   stack<MatrixXd> node_values;  // initialize a stack to hold the node values to use for backpropogation

   // feed forward through the network
   MatrixXd result = input;
   node_values.push(input);
   for (MatrixXd weight : weights)
   {
      result = sigmoid(result * weight, false);
      node_values.push(result);
   }

   // backpropoage the error through the network
   MatrixXd error = target - result;
   for (auto it = weights.rbegin(); it != weights.rend(); ++it)
   {
      MatrixXd end_layer_node_values = node_values.top();
      node_values.pop();
      MatrixXd first_layer_node_values = node_values.top();
      MatrixXd delta = error.array() * sigmoid(end_layer_node_values, true).array();
      error = delta * (*it).transpose();
      *it = *it + alpha * first_layer_node_values.transpose() * delta;
   }
}