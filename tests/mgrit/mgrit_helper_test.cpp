#include "gtest/gtest.h"
#include "mgrit_helper.h"
#include "../test_helper.h"

// setup a fixture to use in the tests below
class MGRITHelperTest : public testing::Test {
 
 protected:

  const phiFuncType mult_2 = [](const weightType& input) 
  {
  	weightType output = input;
	for (int i = 0; i < input.size(); i++)
	{
		output[i] *= 2.0;
	}
	return output;
   };

  const phiFuncType mult_3 = [](const weightType& input) 
  {
  	weightType output = input;
	for (int i = 0; i < input.size(); i++)
	{
		output[i] *= 3.0;
	}
	return output;
   };

   MGRITHelper helper{{mult_2}};
   
   const unsigned int input_size = 50;
   weightType test_weights;
   listOfWeights input;

   // setup test input to be used
   void SetUp() override 
   {

     Matrix<double, 2, 3> test_weights_1;
     Matrix<double, 1, 1> test_weights_2;
	 test_weights_1 << 4.5, 3, 2,
                       8, 1, -.5;
     test_weights_2 << -3;
     test_weights = {test_weights_1, test_weights_2};

	 // initialize the list of weights for the helper tests
	 weightType zeros = {MatrixXd::Zero(2,3), MatrixXd::Zero(1,1)};
	 for (int i = 0; i < input_size; i++)
     {
        i % 3 == 0 ? input.push_back(test_weights) : input.push_back(zeros);
     }

  }

};

TEST_F(MGRITHelperTest, AddArrays){

	// initialize the array to add to the test input and the result array
	listOfWeights array_to_add(input_size, {MatrixXd::Random(2,3), MatrixXd::Random(1,1)});

	// construct the expected output
	listOfWeights expected_output(input_size, {MatrixXd::Zero(2,3), MatrixXd::Zero(1,1)});
	for (int i = 0; i < input_size; i++)
	{
		transform(input[i].begin(), input[i].end(), array_to_add[i].begin(), expected_output[i].begin(), plus<MatrixXd>());
	}

	// check that the value of the expected output and the addOrSubtract function are the same
	listOfWeights result = helper.addOrSubtract(input, array_to_add, plus<MatrixXd>());
	testListOfVectors(result, expected_output);

}

TEST_F(MGRITHelperTest, EuclideanNorm){

	// construct the expected output
	double expected_norm = 0;
	for (int i = 0; i < input_size; i++)
	{
		for (MatrixXd weight : input[i])
		{
			expected_norm += weight.squaredNorm();
		}
	}
	expected_norm = sqrt(expected_norm);

	// check that the value of the expected output and the euclideanNorm function are the same
	double actual_norm = helper.euclideanNorm(input);
	ASSERT_EQ(actual_norm, expected_norm) << "The actual norm " << actual_norm << " is not equal to the expected norm " << expected_norm;

}

TEST_F(MGRITHelperTest, ForwardSolve){

	// initialize the rhs matrix used to solve the linear system
	listOfWeights rhs(input_size, {MatrixXd::Random(2,3), MatrixXd::Random(1,1)});

	// construct the expected output matrix
	listOfWeights expected_output(input_size);
	expected_output[0] = rhs[0];
	vector<phiFuncType> phi = helper.getPhi();
	for (int i = 1; i < input_size; i++)
	{
		transform(rhs[i].begin(), rhs[i].end(), phi[(i-1) % phi.size()](expected_output[i-1]).begin(), back_inserter(expected_output[i]), plus<MatrixXd>());
	}

	// check that the value of the expected output and the forwardSolve function are the same
	listOfWeights actual_output = helper.forwardSolve(rhs);
	testListOfVectors(actual_output, expected_output);

}

TEST_F(MGRITHelperTest, GetPhi){

    // test that the output returned from both functions is the same
    weightType mult_2_output = mult_2(test_weights);
    weightType phi_output = helper.getPhi()[0](test_weights);
    testVectors(phi_output, mult_2_output);

}

TEST_F(MGRITHelperTest, MatMultiply){

	// construct the expected output
	listOfWeights expected_output(input_size);
    expected_output[0] = input[0];
	vector<phiFuncType> phi = helper.getPhi();
    for (int i = 1; i < input_size; i++)
    {
       transform(input[i].begin(), input[i].end(), phi[(i-1) % phi.size()](input[i-1]).begin(), back_inserter(expected_output[i]), minus<MatrixXd>());
    }

	// check that the value of the expected output and the matMultiply function are the same
	listOfWeights actual_output = helper.matMultiply(input);
	testListOfVectors(actual_output, expected_output);

}

TEST_F(MGRITHelperTest, Residual){

	// initialize the rhs matrix to calculate the residuals of the linear system
	listOfWeights rhs(input_size, {MatrixXd::Random(2,3), MatrixXd::Random(1,1)});

	// construct the expected output matrix
    listOfWeights expected_output = helper.addOrSubtract(rhs, helper.matMultiply(input), minus<MatrixXd>());

	// check that the value of the expected output and the residual function are the same
	listOfWeights actual_output = helper.residual(input, rhs);
	testListOfVectors(actual_output, expected_output);

}

TEST_F(MGRITHelperTest, SetPhi){

    // ensure the helper class is initialized properly
    weightType mult_2_output = mult_2(test_weights);
    weightType phi_output = helper.getPhi()[0](test_weights);
    testVectors(phi_output, mult_2_output);

    // give the helper class a new phi and test to make sure the new phi is set correctly
	helper.setPhi({mult_3});
    weightType mult_3_output = mult_3(test_weights);
    phi_output = helper.getPhi()[0](test_weights);
    testVectors(phi_output, mult_3_output);

}