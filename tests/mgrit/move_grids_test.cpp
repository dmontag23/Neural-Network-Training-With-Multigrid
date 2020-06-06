#include "gtest/gtest.h"
#include "move_grids.h"
#include "../test_helper.h"

// setup a fixture to use in the tests below
class MoveGridsTest : public testing::Test {
 
 protected:

  const unsigned int m = 2;
  MoveGrids move{m};

  weightType test_weights;
  listOfWeights input;
  listOfWeights coarse_errors;
  listOfWeights residuals;

  // setup test input to be used
  void SetUp(const unsigned int& input_size) 
  {

    // initialize the weights that will be restricted and projected between different grid levels
    Matrix<double, 2, 3> test_weights_1;
    Matrix<double, 1, 1> test_weights_2;
	  test_weights_1 << 4.5, 3, 2,
                      8, 1, -.5;
    test_weights_2 << -3;
    test_weights = {test_weights_1, test_weights_2};

	  // initialize the list of weights for the projection/restriction tests
	  weightType zeros = {MatrixXd::Zero(2,3), MatrixXd::Zero(1,1)};
	  for (int i = 0; i < input_size; i++)
    {
      i % 3 == 0 ? input.push_back(test_weights) : input.push_back(zeros);
    }

	  // setup random values to be used as coarse errors
	  unsigned int coarse_grid_size =  ceil(static_cast<float>(input_size) / m); 
	  weightType random_weights = {MatrixXd::Random(2,3), MatrixXd::Random(1,1)};
	  for (int i = 0; i < coarse_grid_size; i++)
	  {
		  i % 2 == 0 ? coarse_errors.push_back(random_weights) : coarse_errors.push_back(zeros);
	  }

	  // setup random values to be used as residuals 
	  for (int i = 0; i < input_size; i++)
	  {
		  i % 2 == 0 ? residuals.push_back(random_weights) : residuals.push_back(zeros);
	  }

  }

};

void projectTest(const MoveGrids& move, const listOfWeights& input, const listOfWeights& coarse_errors)
{

    // construct the expected output
    unsigned int m = move.getCoarseningFactor();
    listOfWeights expected_output = input;
	  for (int i = 0; i < coarse_errors.size(); i++)
	  {
 		  transform(expected_output[i*m].begin(), expected_output[i*m].end(), coarse_errors[i].begin(), expected_output[i*m].begin(), plus<MatrixXd>());
	  }

    // run the project function and ensure it is the same as the expected output
    listOfWeights projected_output = move.project(input, coarse_errors);
    testListOfVectors(projected_output, expected_output);

}

void restrictTest(const MoveGrids& move, const listOfWeights& input, const listOfWeights& residuals)
{

	// construct the expected output
	unsigned int m = move.getCoarseningFactor();
  listOfWeights expected_output_grid;
  listOfWeights expected_output_residuals;
	for (int i = 0; i < input.size(); i+=m)
	{
 		expected_output_grid.push_back(input[i]);
 		expected_output_residuals.push_back(residuals[i]);
	}

  // run the restrict function and ensure it is the same as the expected output
  vector<listOfWeights> objects_to_restrict = {input, residuals};
  vector<listOfWeights> coarse_objects = move.restrict(objects_to_restrict);
	listOfWeights coarse_grid = coarse_objects[0];
	listOfWeights coarse_residuals = coarse_objects[1];
  testListOfVectors(coarse_grid, expected_output_grid);
  testListOfVectors(coarse_residuals, expected_output_residuals);

}

TEST_F(MoveGridsTest, GetCoarseningFactor){
    
    unsigned int move_m = move.getCoarseningFactor();
    ASSERT_EQ(move_m, m) << "The coarsening factor " << move_m << " from the move class is not equal to " << m;

}

TEST_F(MoveGridsTest, ProjectEvenSize){
	
	SetUp(10);
	projectTest(move, input, coarse_errors);

}

TEST_F(MoveGridsTest, ProjectOddSize){
	
	SetUp(17);
	projectTest(move, input, coarse_errors);

}

TEST_F(MoveGridsTest, RestrictEvenSize){
	
	SetUp(10);
	restrictTest(move, input, residuals);

}

TEST_F(MoveGridsTest, RestrictOddSize){
	
	SetUp(17);
	restrictTest(move, input, residuals);

}

TEST_F(MoveGridsTest, SetCoarseningFactor){
    
    // ensure the class is initialized properly
    unsigned int move_m = move.getCoarseningFactor();
    ASSERT_EQ(move_m, m) << "The coarsening factor " << move_m << " from the move class is not equal to " << m;
    
    // give the class a new coarsening factor and test to make sure the new coarsening factor is set correctly
    unsigned int new_m = 4;
    move.setCoarseningFactor(new_m);
    move_m = move.getCoarseningFactor();
    ASSERT_EQ(move_m, new_m) << "The coarsening factor " << move_m << " from the move class is not equal to the new m " << new_m;

}