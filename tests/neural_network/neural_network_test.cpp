#include "gtest/gtest.h"
#include "neural_network.h"
#include "../test_helper.h"

// setup a fixture to use in the tests below
class NeuralNetworkTest : public testing::Test {
 
 protected:

  const float alpha = 2.0;
  const MatrixXd random1 = MatrixXd::Random(3,4);
  const MatrixXd random2 = MatrixXd::Random(4,1);
  const weightType weights = {random1, random2};

  NeuralNetwork nn1{alpha, weights};

};

TEST_F(NeuralNetworkTest, GetAlpha){

    float nn_alpha = nn1.getAlpha();
    ASSERT_EQ(nn_alpha, alpha) << "The NN alpha " << nn_alpha << " is not equal to " << alpha;

}

TEST_F(NeuralNetworkTest, GetWeights){

    weightType nn_weights = nn1.getWeights();
    testVectors(nn_weights, weights);

}

TEST_F(NeuralNetworkTest, SetAlpha){
    
    // ensure the nn is initialized properly
    float nn_alpha = nn1.getAlpha();
    ASSERT_EQ(nn_alpha, alpha) << "The NN alpha " << nn_alpha << " is not equal to " << alpha;
    
    // give the nn a new alpha and test to make sure the new alpha is set correctly
    float new_alpha = 4.26432;
    nn1.setAlpha(new_alpha);
    nn_alpha = nn1.getAlpha();
    ASSERT_EQ(nn_alpha, new_alpha) << "The new NN alpha " << nn_alpha << " is not equal to " << new_alpha;

}

TEST_F(NeuralNetworkTest, SetWeights){

    // ensure the nn is initialized properly
    weightType nn_weights = nn1.getWeights();
    testVectors(nn_weights, weights);

    // give the nn new weights and test to make sure the new weights are set correctly
    MatrixXd ones = MatrixXd::Ones(10,2);
    weightType new_weights = {ones, ones, ones};
    nn1.setWeights(new_weights);
    nn_weights = nn1.getWeights();
    testVectors(nn_weights, new_weights);

}

TEST_F(NeuralNetworkTest, TrainBatchSigmoidActivation){
    
    // ensure the nn is initialized properly
    weightType nn_weights = nn1.getWeights();
    testVectors(nn_weights, weights);

    // define the sigmoid activation function and its derivative
    auto sigmoid = [](const MatrixXd& input) { return 1.0 / (1.0 + (-1.0 * input.array()).exp()); };
    auto sigmod_deriv = [](const MatrixXd& input) { return input.array() * (1.0 - input.array()); };

    // process the input through the nn using the sigmoid activation function
    MatrixXd input = MatrixXd::Ones(10,3);                         
    MatrixXd hidden1 = input * random1;                   
    hidden1 = sigmoid(hidden1);        
    MatrixXd output = hidden1 * random2;                 
    output = sigmoid(output);   

    // get the errors
    MatrixXd target = MatrixXd::Ones(10,1);   
    MatrixXd errors = target - output;   

    // backpropagate the error through the network                    
    MatrixXd delta2 = errors.array() * sigmod_deriv(errors);     
    MatrixXd error2 = delta2 * random2.transpose();                                  
    MatrixXd delta1 = error2.array() * sigmod_deriv(hidden1);  

    // update weights
    MatrixXd new_weights1 = random1 + alpha * input.transpose()*delta1;               
    MatrixXd new_weights2 = random2 + alpha * hidden1.transpose()*delta2;
    weightType new_weights = {new_weights1, new_weights2};

    // train the nn above using the class method and check that the two values are the same
    nn1.train(input, target);
    nn_weights = nn1.getWeights();
    testVectors(nn_weights, new_weights);

}