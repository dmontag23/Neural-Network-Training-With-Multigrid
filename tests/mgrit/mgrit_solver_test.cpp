#include "gtest/gtest.h"
#include "mgrit_solver.h"
#include "neural_network.h"
#include "../test_helper.h"

// setup a fixture to use in the tests below
class MGRITSolverTest : public testing::Test {
 
 protected:

  const unsigned int m = 2;
  const unsigned int max_level = 2;
  const unsigned int input_size = 100;

  weightType test_weights;
  listOfWeights input{input_size + 1, {MatrixXd::Zero(3,4), MatrixXd::Zero(4,1)}};

  Matrix<double, 4, 3> nn_input;
  Matrix<double, 4, 1> target;
  NeuralNetwork testNN = {1.0, test_weights};

  const phiFuncType phi1 = [this](const weightType &weights) 
  {
  	testNN.setAlpha(1.0);
    testNN.setWeights(weights);
    testNN.train(nn_input, target);
    return testNN.getWeights();
  };

  const phiFuncType phi2 = [this](const weightType &weights) 
  {
  	testNN.setAlpha(2.0);
    testNN.setWeights(weights);
    testNN.train(nn_input, target);
    return testNN.getWeights();
  };

  MGRITSolver solver1{m, {{phi1}, {phi2}}, max_level};
  const MGRITHelper helper1{{phi1}};

  // setup test input to be used
  void SetUp() override 
  {

	// initialize the weights for the relaxation tests
	test_weights = {MatrixXd::Random(3,4), MatrixXd::Random(4,1)};
	input[0] = test_weights;

	// initialize the data for the neural network
    nn_input << 0.0, 0.0, 1.0,
                0.0, 1.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 1.0;

    target << 0.0,
  			  1.0,
  			  1.0,
  			  0.0;

	testNN.setWeights(test_weights);

  }

};


listOfWeights trainNN(NeuralNetwork nn, const unsigned int& training_steps, const MatrixXd& nn_input, const MatrixXd& target)
{

    listOfWeights result;
    result.push_back(nn.getWeights());
  	for (int i = 0; i < training_steps; i++)
  	{
  		nn.train(nn_input, target);
  		result.push_back(nn.getWeights());
  	}
    return result;
}

void testRun(const NeuralNetwork& nn, const MatrixXd& nn_input, const MatrixXd& nn_target, MGRITSolver solver1, const MGRITHelper& helper1, const listOfWeights& input, const bool& f_iteration)
{

	// construct the expected output
	unsigned int input_size = input.size() - 1;
    listOfWeights expected_output = trainNN(nn, input_size, nn_input, nn_target);

    // setup the MGRIT algorithm
    // construct the rhs
    listOfWeights rhs = input;

    // set the tolerance to use for convergence
    double tol = pow(10, -9) * sqrt(input_size + 1);
    listOfWeights actual_output = solver1.run(input, rhs, tol, f_iteration);

    // compute the final residuals to ensure they are below the tolerance
    listOfWeights final_residuals = helper1.addOrSubtract(rhs, helper1.matMultiply(actual_output), minus<MatrixXd>());
    double residual_norm = helper1.euclideanNorm(final_residuals);
    ASSERT_TRUE(residual_norm < tol) << "The norm of the residual " << residual_norm << " is not less than " << tol;    

    // ensure the actual and expected arrays are close to the same by checking their norms
    double diff_norm = abs(helper1.euclideanNorm(actual_output) - helper1.euclideanNorm(actual_output));
    ASSERT_TRUE(diff_norm < tol) << "The difference of the norms between the actual and expected output " << diff_norm << " is not less than " << tol;

}

TEST_F(MGRITSolverTest, GetCoarseningFactor){
    
    unsigned int solver_m = solver1.getCoarseningFactor();
    ASSERT_EQ(solver_m, m) << "The coarsening factor " << solver_m << " from the MGRITSolver class is not equal to " << m;

}

TEST_F(MGRITSolverTest, GetDisplayStats){
    
    bool display_stats = solver1.getDisplayStats();
    ASSERT_FALSE(display_stats) << "The display stats are initialized to " << display_stats;

}

TEST_F(MGRITSolverTest, GetMaxLevel){
    
    unsigned int level = solver1.getMaxLevel();
    ASSERT_EQ(level, max_level) << "The max level " << level << " from the MGRITSolver class is not equal to " << max_level;

}

TEST_F(MGRITSolverTest, GetPhis){

    // test that the output returned from both functions is the same
    weightType phi_1_expected_output = phi1(test_weights);
    weightType phi_2_expected_output = phi2(test_weights);
    vector<vector<phiFuncType>> phis = solver1.getPhis();
    weightType phi_1_actual_output = phis[0][0](test_weights);
    weightType phi_2_actual_output = phis[1][0](test_weights);
    testVectors(phi_1_actual_output, phi_1_expected_output);
    testVectors(phi_2_actual_output, phi_2_expected_output);

}

TEST_F(MGRITSolverTest, SetCoarseningFactor){
    
    // ensure the class is initialized properly
    unsigned int m = solver1.getCoarseningFactor();
    ASSERT_EQ(m, m) << "The coarsening factor " << m << " from the MGRITSolver class is not equal to " << m;
    
    // give the class a new coarsening factor and test to make sure the new coarsening factor is set correctly
    unsigned int new_m = 4;
    solver1.setCoarseningFactor(new_m);
    m = solver1.getCoarseningFactor();
    ASSERT_EQ(m, new_m) << "The coarsening factor " << m << " from the MGRITSolver class is not equal to the new " << new_m;

}

TEST_F(MGRITSolverTest, SetDisplayStats){
    
    // ensure the class is initialized properly
    bool display_stats = solver1.getDisplayStats();
    ASSERT_FALSE(display_stats) << "The display stats are initialized to " << display_stats;
    
    // change the class to display stats
    solver1.setDisplayStats(true);
    display_stats = solver1.getDisplayStats();
    ASSERT_TRUE(display_stats) << "The display stats are set to " << display_stats;

}

TEST_F(MGRITSolverTest, SetMaxLevel){
    
    // ensure the class is initialized properly
    unsigned int level = solver1.getMaxLevel();
    ASSERT_EQ(level, max_level) << "The max level " << level << " from the MGRITSolver class is not equal to " << max_level;
    
    // give the class a new max level and test to make sure the new max level is set correctly
    unsigned int new_max_level = 4;
    solver1.setMaxLevel(new_max_level);
    level = solver1.getMaxLevel();
    ASSERT_EQ(level, new_max_level) << "The max level " << level << " from the MGRITSolver class is not equal to " << new_max_level;

}

TEST_F(MGRITSolverTest, SetPhis){

    // ensure the MGRITSolver class is initialized properly
    weightType phi_1_expected_output = phi1(test_weights);
    weightType phi_2_expected_output = phi2(test_weights);
    vector<vector<phiFuncType>> phis = solver1.getPhis();
    weightType phi_1_actual_output = phis[0][0](test_weights);
    weightType phi_2_actual_output = phis[1][0](test_weights);
    testVectors(phi_1_actual_output, phi_1_expected_output);
    testVectors(phi_2_actual_output, phi_2_expected_output);

    // give the MGRITSolver class a new set of phis and test to make sure the new phis are set correctly
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

	phi_1_expected_output = mult_2(test_weights);
	phi_2_expected_output = mult_3(test_weights);

	vector<vector<phiFuncType>> new_phis = {{mult_2}, {mult_3}};
	solver1.setPhis(new_phis);
	phis = solver1.getPhis();
    phi_1_actual_output = phis[0][0](test_weights);
    phi_2_actual_output = phis[1][0](test_weights);
    testVectors(phi_1_actual_output, phi_1_expected_output);
    testVectors(phi_2_actual_output, phi_2_expected_output);

}

TEST_F(MGRITSolverTest, VIteration2Levels){

	solver1.setMaxLevel(2);
	testRun(testNN, nn_input, target, solver1, helper1, input, false);

}

TEST_F(MGRITSolverTest, VIteration10Levels){

    const unsigned int max_level = 10;
	vector<vector<phiFuncType>> phis(max_level, {phi1});
    solver1.setMaxLevel(max_level);
    solver1.setPhis(phis);
	testRun(testNN, nn_input, target, solver1, helper1, input, false);

}

TEST_F(MGRITSolverTest, FIteration2Levels){

	solver1.setMaxLevel(2);
	testRun(testNN, nn_input, target, solver1, helper1, input, true);

}

TEST_F(MGRITSolverTest, FIteration10Levels){

    const unsigned int max_level = 10;
	vector<vector<phiFuncType>> phis(max_level, {phi1});
    solver1.setMaxLevel(max_level);
    solver1.setPhis(phis);
	testRun(testNN, nn_input, target, solver1, helper1, input, true);

}